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


class SelfLoss(nn.Module):
    def __init__(self, stud_dist, stud_uncert_as_a_fraction_of_depth, kldiv, teacher_dist):
        super().__init__()

        if kldiv:
            if stud_dist == 'normal' and teacher_dist == 'normal':
                self.loss = self.normal_normal_kldiv
            elif stud_dist == 'laplace' and teacher_dist == 'laplace':
                self.loss = self.laplace_laplace_kldiv
            else:
                raise NotImplementedError

        if stud_dist == 'laplace':
            self.loss = self.laplace_nll
        elif stud_dist == 'normal':
            self.loss = self.normal_nll
        else:
            raise NotImplementedError
        
        if stud_uncert_as_a_fraction_of_depth:
            self.log_pred_uncert = self.rel_uncert
        else:
            self.log_pred_uncert = self.abs_uncert


    def abs_uncert(self, uncert):
        return uncert, torch.exp(uncert)
    
    def rel_uncert(self, uncert):
        return torch.log(uncert), uncert


    def laplace_nll(self, pred, pred_uncert, teacher_depth, teacher_uncert):
        log_pred_uncert, pred_uncert = self.log_pred_uncert(pred_uncert)
        return torch.abs(pred - teacher_depth) / pred_uncert + log_pred_uncert

    def normal_nll(self, pred, pred_uncert, teacher_depth, teacher_uncert):
        log_pred_uncert, pred_uncert = self.log_pred_uncert(pred_uncert)
        return 0.5 * ((pred - teacher_depth) / pred_uncert) ** 2 + log_pred_uncert
    
    def normal_normal_kldiv(self, pred, pred_uncert, teacher_depth, teacher_uncert):
        log_pred_uncert, pred_uncert = self.log_pred_uncert(pred_uncert)
        log_teacher_uncert, teacher_uncert = self.log_pred_uncert(teacher_uncert)
        return log_teacher_uncert  - log_pred_uncert + (pred_uncert ** 2 + (pred - teacher_depth) ** 2) / (2 * teacher_uncert ** 2)
    
    def laplace_laplace_kldiv(self, pred, pred_uncert, teacher_depth, teacher_uncert):
        log_pred_uncert, pred_uncert = self.log_pred_uncert(pred_uncert)
        log_teacher_uncert, teacher_uncert = self.log_pred_uncert(teacher_uncert)
        return log_teacher_uncert  - log_pred_uncert + (pred_uncert * torch.exp(-(pred - teacher_depth).abs() / pred_uncert) + (pred - teacher_depth).abs()) / teacher_uncert
    
    def forward(self, pred, pred_uncert, teacher_depth, teacher_uncert):
        return self.loss(pred, pred_uncert, teacher_depth, teacher_uncert).mean()

    
