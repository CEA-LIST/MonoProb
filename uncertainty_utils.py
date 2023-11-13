#
# MIT License
#
# Copyright (c) 2023 Rémi Marsal remi.marsal@cea.fr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import cv2

from layers import *

# Generation of the samples
# "Offsets" should be understood as offsets relative to the mean of the distribution
class OffsetsSampling(nn.Module):
    def __init__(self, sample_size, uncert_as_a_fraction_of_depth=False, distribution='normal') -> None:
        """Initialization of the offsets"""
        super().__init__()
        assert sample_size % 2 == 1, 'sample_size should be odd'
        self.sample_size = sample_size

        # The ratios between the density of samples and the density of the mean depth are set
        ratios = torch.arange(1, sample_size // 2 + 1, device=torch.device('cuda')) / (sample_size // 2 + 1)
        
        # The weight of each sample is computed
        probas = torch.cat([ratios, torch.arange(1, 2, device=torch.device('cuda')), ratios.flip(dims=[0])])
        self.probas = probas / probas.sum()

        # Converting ratios into offsets
        if distribution == 'normal':
            neighbors = (-2 * torch.log(ratios)).sqrt()
        elif distribution == 'laplace':
            neighbors = -torch.log(ratios) / np.sqrt(2)

        if uncert_as_a_fraction_of_depth:
            self.offsets_multiplier = lambda x: x
        else:
            self.offsets_multiplier = lambda x: 1.

        self.offsets = torch.cat([-neighbors, torch.arange(1, device=torch.device('cuda')), neighbors.flip(dims=[0])])
        self.get_offsets = self.normal_laplace_offsets
    
    def normal_laplace_offsets(self, uncerts: Tensor, means):
        """Scale offsets for a Laplace or Normal distribution"""
        dims = uncerts.size()
        return self.offsets_multiplier(means) * uncerts * self.offsets.view(*(1,)*len(dims[:-3]), self.sample_size, 1, 1)
    
    def forward(self, means, uncerts):

        # Get samples
        samples = means + self.get_offsets(uncerts, means)

        # Expand the weights to get a map of weights
        samples_dims = samples.size()
        probas = self.probas.view(*(1,)*len(samples_dims[:-3]), self.sample_size, *(1,)*len(samples_dims[-2:])).expand_as(samples)
        return samples, probas.unsqueeze(-3)


class BackprojectDepthUncertainty(nn.Module):
    """Layer to transform a map of depth samples into point clouds
    """
    def __init__(self, batch_size, height, width, sample_size):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.sample_size = sample_size

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.ones_samlples = nn.Parameter(torch.ones(self.batch_size, 1, self.sample_size * self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = cam_points.unflatten(dim=-1, sizes=(self.height, self.width)).unsqueeze(-3).expand(-1, -1, self.sample_size, -1, -1).flatten(start_dim=-3)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones_samlples], 1)

        return cam_points


class Project3DUncertainty(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7, sample_size=1, mask_out=None):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.sample_size = sample_size
        self.mask = self.mask_out_samples
    
    def mask_out_samples(self, coords, margin=0.):
        return torch.logical_and(
               torch.logical_and(coords[:, 0] >= margin, coords[:, 0] <= self.width-1-margin),
               torch.logical_and(coords[:, 1] >= margin, coords[:, 1] <= self.height-1-margin)).unsqueeze(2)

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.sample_size, self.height, self.width)
        mask = self.mask(pix_coords)
        pix_coords = pix_coords.permute(0, 2, 3, 4, 1).reshape(self.batch_size * self.sample_size, self.height, self.width, 2)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords, mask


def remove_out_distributions(probas, mask, sample_size):
    mask_any = mask.prod(dim=1, keepdim=True)
    probas_updates = probas * mask_any
    probas_updates[:, sample_size//2:sample_size//2+1] += (1-mask_any).float()
    return probas_updates


def remove_out_samples(probas, mask, sample_size):
    probas = probas.clone()
    
    for i  in range(sample_size//2):
        mask_samples = (mask[:, i] * mask[:, sample_size-1-i]).unsqueeze(1)
        probas[:, :i+1] *= mask_samples
        probas[:, sample_size-1-i:] *= mask_samples
    
    mask_samples = mask[:, sample_size//2].unsqueeze(1)
    probas[:, :sample_size//2] *= mask_samples
    probas[:, sample_size//2+1:] *= mask_samples
    return (probas / probas.sum(1, keepdim=True))


def no_probas_updates(probas, mask, sample_size):
    return probas


def save_uncertainty_visualization(gt, pred, std, mask, index, path, im=None):
    """Generate and save qualitative results from predictions and ground truth 
    including depth, errors and uncertainty map
    """
    visu_dict = {}

    disp = 1/pred
    visu_dict['disp'] = 255 * (disp) / np.max(disp)

    visu_dict['abs_std'] = np.clip(std, 0, np.max(std * mask))
    visu_dict['abs_std'] = visu_dict['abs_std'] / np.max(visu_dict['abs_std']) * 255

    visu_dict['abs_std_mask'] = visu_dict['abs_std'] * mask
    
    visu_dict['abs_error'] = np.abs(gt - pred) * mask
    visu_dict['abs_error'] = visu_dict['abs_error'] / (np.max(visu_dict['abs_error'])) * 255

    if im is not None:
        visu_dict['im'] = im * 255

    for key, value in visu_dict.items():
        if key == 'depth':
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_MAGMA))
        elif 'error' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_HOT))
        elif 'std' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_HOT))
        elif 'im' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.cvtColor(value.astype('uint8'), cv2.COLOR_BGR2RGB))
        elif 'disp' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_MAGMA))
        else:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_JET))


def save_visualization(gt, pred, mask, index, path, im=None):
    """Generate and save qualitative results from predictions and ground truth 
    including depth and errors
    """
    visu_dict = {}

    disp = 1/pred
    visu_dict['disp'] = 255 * (disp) / np.max(disp)
    
    visu_dict['abs_error'] = np.abs(gt - pred) * mask
    visu_dict['abs_error'] = visu_dict['abs_error'] / (np.max(visu_dict['abs_error'])) * 255

    if im is not None:
        visu_dict['im'] = im * 255

    for key, value in visu_dict.items():
        if key == 'depth':
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_MAGMA))
        elif 'error' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_HOT))
        elif 'std' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_HOT))
        elif 'im' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.cvtColor(value.astype('uint8'), cv2.COLOR_BGR2RGB))
        elif 'disp' in key:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_MAGMA))
        else:
            cv2.imwrite(os.path.join(path, str(index) + '_' + key + '.png'), cv2.applyColorMap(value.astype('uint8'), cv2.COLORMAP_JET))