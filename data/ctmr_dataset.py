import os
from collections import defaultdict
from time import time
from data.multi_modal_dataset import MultiModalDataset
from PIL import Image


class CTMRDataset(MultiModalDataset):

    def __init__(self, opt):

        if opt.isTrain:
            self.dataset_root = opt.dataroot
        else:
            self.dataset_root = opt.test_dataroot

        self.mr_path = os.path.join(self.dataset_root, 'mr')

        super(CTMRDataset, self).__init__(opt)


    def get_modal_names(self):
        return ['ct', 'mr']


    def load_data(self):
        self.n_data = len(os.listdir(self.mr_path))
        print('Loading {} Dataset with "{}" mode...'.format(self.dataset, self.mode))
        start = time()
        print('load data from raw')
        data_dict = defaultdict(list)
        for index in range(self.n_data):
            for modal in self.modal_names:
                dir = os.path.join(self.dataset_root, modal)
                ig = os.listdir(dir)
                img = Image.open(os.path.join(dir, ig[index])).convert('RGB')
                data_dict[modal].append(img)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))
        return data_dict

