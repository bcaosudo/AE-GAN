from collections import OrderedDict

import torch
from torch import nn
from .base_model import BaseModel
from . import networks

class AEGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='mh_resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.n_input_modal = opt.n_input_modal
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'SR_L1', 'G_SR']
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        self.netG = networks.define_MHG(opt.n_input_modal, opt.input_nc+opt.n_input_modal+1, opt.output_nc, opt.ngf, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionCls = nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.criterionL2 = torch.nn.MSELoss()
            self.criterionKL = torch.nn.KLDivLoss()

        self.all_modal_names = opt.modal_names
        self.sr_weight = opt.sr_weight
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def set_input(self, input):

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_no_mask = input['B'][:, :self.opt.input_nc].to(self.device)
        self.modal_names = [i[0] for i in input['modal_names']]

        target_modal_names = input['modal_names'][-1]
        self.real_B_Cls = torch.tensor([self.all_modal_names.index(i) for i in target_modal_names]).to(self.device)

        if hasattr(self, 'sr'):
            self.sr.real_A = self.real_B_no_mask
            self.sr.real_B = self.real_B_no_mask

    def forward(self, train=False):
        if train:
            self.fake_B, self.decoder_features = self.netG(self.real_A, True)
        else:
            self.fake_B = self.netG(self.real_A, train)  # G(A)


    def backward_D(self):

        fake_B = self.fake_B.detach() # fake_B from generator
        _, g_pred_fake, _ = self.sr.netG(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(g_pred_fake, False)

        fake_rec_B = self.sr.fake_B.detach()  # fake_B from autoencoder
        _, e_pred_fake, _ = self.sr.netG(fake_rec_B)
        self.loss_D_fake += self.criterionGAN(e_pred_fake, False)
        # Real
        _, pred_real, _ = self.sr.netG(self.real_B_no_mask)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_SR_L1 = self.sr.compute_loss()
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake * 0.5 + self.loss_D_real) * 0.5 + self.loss_SR_L1

        self.loss_D.backward()

    def backward_G(self):
        _, g_pred_fake, _ = self.sr.netG(self.fake_B) 
        self.loss_G_GAN = self.criterionGAN(g_pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B_no_mask) * self.opt.lambda_L1
        self.loss_G_SR = 0
        sr_decoder_features = self.sr.get_features()
        for i in range(len(sr_decoder_features)):
            self.loss_G_SR += self.criterionKL(torch.nn.functional.log_softmax(self.decoder_features[i]), torch.nn.functional.softmax(sr_decoder_features[i])) * self.sr_weight
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_SR
        self.loss_G.backward()

    def optimize_parameters(self):
        self.set_requires_grad(self.sr.netG, True)  # enable backprop for D
        self.sr.forward()
        self.forward(True)                   # compute fake images: G(A)

        self.optimizer_SR.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_SR.step()          # update D's weights
        self.set_requires_grad(self.sr.netG, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_visuals(self):
        """Calculate additional output images for tensorboard visualization"""
        pass

    def get_current_visuals(self):
        modal_imgs = []
        for i in range(self.n_input_modal):
            modal_imgs.append(self.real_A[:, i*(self.n_input_modal+1+self.opt.input_nc):i*(self.n_input_modal+1+self.opt.input_nc)+self.opt.input_nc, :, :])
        modal_imgs.append(self.real_B_no_mask)
        visual_ret = OrderedDict()
        for name, img in zip(self.modal_names, modal_imgs):
            visual_ret[name] = img
        visual_ret['fake_' + self.modal_names[-1]] = self.fake_B
        if hasattr(self, 'sr'):
            visual_ret['reconstruct'] = self.sr.fake_B
        return visual_ret

    def add_srmodel(self, sr_model):
        self.sr = sr_model
        self.optimizer_SR = torch.optim.Adam(self.sr.netG.parameters(), lr=self.opt.lr,
                                            betas=(self.opt.beta1, 0.999))
