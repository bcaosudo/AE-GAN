from copy import copy
import torch


def create_modal_mask(modal_names, size=256):
    n_modal = len(modal_names)
    modal_mask_dict = {}
    for i, modal in enumerate(modal_names):
        mask = torch.zeros([n_modal, size, size])
        mask[i, :, :] = 1
        modal_mask_dict[modal] = mask
    return modal_mask_dict


def permute_modal_names(modal_names):
    modal_permutations = []
    for i in range(len(modal_names)):
        modal_permutations.append(copy(modal_names))
        first_modal = modal_names.pop(0)
        modal_names.append(first_modal)
    return modal_permutations