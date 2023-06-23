import os
from options.config import load_config
from data import create_dataset
from models import create_model
from util import util
import numpy as np


if __name__ == '__main__':
    opt = load_config()
    opt.load_size = 256
    opt.results_dir = 'results/'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.isTrain = False
    phase = 'val' if not opt.isTrain else 'train'
    dataloader = create_dataset(opt)
    opt.n_input_modal = dataloader.dataset.n_modal - 1
    opt.modal_names = dataloader.dataset.get_modal_names()
    n_modal = 1 if 'encoder' in opt.name or 'pix2pix' in opt.name or 'cycle' in opt.name else dataloader.dataset.n_modal
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    dst_dir = os.path.join(opt.results_dir, opt.name, phase + '-' + str(opt.epoch))
    os.makedirs(dst_dir, exist_ok=True)

    for i, data in enumerate(dataloader):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference: forward() and compute_visuals()
        visuals = model.get_current_visuals()  # get image results
        imgs = []
        labels = []
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            imgs.append(image_numpy)
            labels.append(label)
            label_dst_dir = os.path.join(dst_dir, label)
            os.makedirs(label_dst_dir, exist_ok=True)
            util.save_image(image_numpy, os.path.join(label_dst_dir, '{}.jpg'.format(i//n_modal + 1)))
        cat_img = np.concatenate(imgs, axis=1)
        cat_dir = os.path.join(dst_dir, '-'.join(labels))
        os.makedirs(cat_dir, exist_ok=True)
        util.save_image(cat_img, os.path.join(cat_dir, '{}.jpg'.format(i//n_modal+1)))