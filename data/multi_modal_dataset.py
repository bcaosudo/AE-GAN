import os
import pickle
from collections import defaultdict
from time import time
import torch
from data.utils import permute_modal_names, create_modal_mask
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image


class MultiModalDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = opt.dataroot
        self.mode = opt.dataset_mode
        self.dataset = opt.dataset
        self.isTrain = opt.isTrain
        self.isGray = opt.input_nc == 1
        self.phase = 'trian' if opt.isTrain else 'val'
        # 1. load data
        self.modal_names = self.get_modal_names()
        self.data_dict = self.load_data()

        # 2. create modal mask
        self.n_modal = len(self.modal_names)
        self.modal_mask_dict = create_modal_mask(self.modal_names)

        if self.mode == 'all':
            self.modal_permutations = permute_modal_names(self.modal_names)
        elif self.mode == 'same':
            self.modal_permutations = self.modal_names
        else:
            self.source = self.opt.source
            self.dst = self.opt.dst
            self.modal_permutations = [self.opt.source]


    def load_data(self):
        datapath_dict = self.load_img_paths()
        self.n_data = len(datapath_dict[self.modal_names[0]])
        print('Loading {} Dataset with "{}" mode...'.format(self.dataset, self.mode))
        start = time()
        cache_path = os.path.join(self.dataroot, self.phase + '.pkl')
        if os.path.exists(cache_path):
            print('load data cache from: ', cache_path)
            with open(cache_path, 'rb') as fin:
                data_dict = pickle.load(fin)
        else:
            print('load data from raw')
            data_dict = defaultdict(list)
            for index in range(self.n_data):
                for modal in self.modal_names:
                    img = Image.open(datapath_dict[modal][index]).convert('RGB')
                    data_dict[modal].append(img)
            with open(cache_path, 'wb') as fin:
                pickle.dump(data_dict, fin)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))
        return data_dict

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # A for input; B for target
        if self.mode == 'all':
            modal_order = self.modal_permutations[index % len(self.modal_permutations)]
            input_modal_names = modal_order[:-1]
            target_modal_name = modal_order[-1]

            B = self.data_dict[target_modal_name][index // len(self.modal_permutations)]
            transform_params = get_params(self.opt, B.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=self.isGray)
            B_transform = get_transform(self.opt, transform_params, grayscale=self.isGray)

            B = B_transform(B)

            A = []
            target_mask = self.modal_mask_dict[target_modal_name]
            for input_modal in input_modal_names:
                A_img = self.data_dict[input_modal][index // len(self.modal_permutations)]
                A_img = A_transform(A_img)
                A_img = torch.cat([A_img, target_mask])
                A.append(A_img)

            return {'A': torch.cat(A), 'B': B, 'modal_names':modal_order}

        elif self.mode == 'same':
            modal_name = self.modal_permutations[index % len(self.modal_permutations)]
            modal_input = self.data_dict[modal_name][index // len(self.modal_permutations)]
            transform_params = get_params(self.opt, modal_input.size)
            transform = get_transform(self.opt, transform_params, grayscale=self.isGray)
            modal_input = transform(modal_input)
            input = {
                'A': modal_input,
                'B': modal_input,
                'modal_name': modal_name
            }
            return input

        elif self.mode == 'single':
            modal_input = self.data_dict[self.source][index]
            modal_target = self.data_dict[self.dst][index]
            transform_params = get_params(self.opt, modal_input.size)
            transform = get_transform(self.opt, transform_params, grayscale=self.isGray)
            modal_input = transform(modal_input)
            modal_target = transform(modal_target)
            input = {
                'A': modal_input,
                'B': modal_target
            }
            return input

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_data * len(self.modal_permutations)

    def load_img_paths(self):
        pass

    def get_modal_names(self):
        pass