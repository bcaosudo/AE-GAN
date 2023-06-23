import argparse
import os
from easydict import EasyDict as edict
import yaml


def load_config(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description='Train GANs')
        parser.add_argument('config', help='train config file path')
        args = parser.parse_args()
        config_path = args.config

    with open(config_path) as fin:
        opt = yaml.load(fin, Loader=yaml.FullLoader)
        opt = edict(opt)
    if 'auto_encoder' not in config_path:
        for k, v in opt.items():
            print(k, ':', v)

    opt.name = opt.dataset + '/' + opt.name
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    return opt
