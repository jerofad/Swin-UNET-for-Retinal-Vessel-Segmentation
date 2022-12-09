
""" Consistency loss from https://arxiv.org/pdf/2205.15428.pdf
"""


import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConsistencyTrainingLoss(nn.Module):

    def __init__(self, adaptive=True):
        super(ConsistencyTrainingLoss, self).__init__()
        self.epsilon = 1e-5
        self.adaptive = adaptive
        self.jaccard = vanilla_losses.JaccardLoss()

    def forward(self, new_mask, old_mask, new_seg, old_seg, iou_weight=None):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        vanilla_jaccard = vanilla_losses.JaccardLoss()(old_seg, old_mask)

        sil = torch.sum(
            difference(
                difference(new_mask, old_mask),
                difference(new_seg, old_seg))
        ) / torch.sum(torch.clamp(new_mask + old_mask + new_seg + old_seg, 0, 1) + self.epsilon)

        return (1 - iou_weight) * vanilla_jaccard + iou_weight * sil


class SIL(nn.Module):
    def __init__(self):
        super(SIL, self).__init__()
        self.epsilon = 1e-5

    def forward(self, new_mask, old_mask, new_seg, old_seg):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        perturbation_loss = torch.sum(
            difference(
                difference(new_mask, old_mask),
                difference(new_seg, old_seg))
        ) / torch.sum(torch.clamp(new_mask + old_mask + new_seg + old_seg, 0, 1) + self.epsilon)  # normalizing factor
        
        return perturbation_loss
