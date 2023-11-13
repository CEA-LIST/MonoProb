#
# MIT License
#
# Copyright (c) 2020 Matteo Poggi m.poggi@unibo.it
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


def compute_errors_uncertainty(gt, pred, std):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = thresh < 1.25

    abs_diff = np.abs(gt - pred)
    sq_diff = abs_diff ** 2

    rmse = np.sqrt(sq_diff.mean())

    abs_rel = abs_diff / gt
    abs_uncert = np.abs(abs_diff - std)

    rel_unc = abs_uncert / gt
    rmse_unc = np.sqrt((abs_uncert ** 2).mean())

    aucs = compute_aucs(gt, pred, std)
    list_aucs = [{'AUSE_' + key: value[0], 'AURG_' + key: value[1]} for key, value in aucs.items()]
    aucs = {}
    for el in list_aucs:
        aucs.update(el)

    return {
        'abs_rel': abs_rel.mean(), 
        'rmse': rmse,
        'a1': a1.astype(float).mean(),
    } | aucs | { 
        'rel_u': rel_unc.mean(), 
        'rmse_u': rmse_unc,
    }

def compute_errors_for_auc(gt, pred, metrics, mask=None, reduce_mean=False):
    """Revised compute_eigen_errors function used for uncertainty metrics, with optional reduce_mean argument and (1-a1) computation
    """
    results = []
    
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    
    if "abs" in metrics:
        abs_rel = (np.abs(gt - pred) / gt)
        if reduce_mean:
            abs_rel = abs_rel.mean()
        results.append(abs_rel)

    if "rmse" in metrics:
        rmse = (gt - pred) ** 2
        if reduce_mean:
            rmse = np.sqrt(rmse.mean())
        results.append(rmse)

    if "a1" in metrics:
        a1 = np.maximum((gt / pred), (pred / gt))
        if reduce_mean:
        
            # invert to get outliers
            a1 = (a1 >= 1.25).astype(float).mean()
        results.append(a1)

    return results


def compute_aucs(gt, pred, uncert, intervals=50):
    """Computation of auc metrics
    """

    uncertainty_metrics = ["abs", "rmse", "a1"]
    
    # results dictionaries
    AUSE = {"abs":0, "rmse":0, "a1":0}
    AURG = {"abs":0, "rmse":0, "a1":0}

    # revert order (high uncertainty first)
    uncert = -uncert
    true_uncert = compute_errors_for_auc(gt,pred, uncertainty_metrics)
    true_uncert = {"abs":-true_uncert[0],"rmse":-true_uncert[1],"a1":-true_uncert[2]}

    # prepare subsets for sampling and for area computation
    quants = [100./intervals*t for t in range(0,intervals)]
    plotx = [1./intervals*t for t in range(0,intervals+1)]

    # get percentiles for sampling and corresponding subsets
    thresholds = [np.percentile(uncert, q) for q in quants]
    subs = [(uncert >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    sparse_curve = {m:[compute_errors_for_auc(gt,pred,metrics=[m],mask=sub,reduce_mean=True)[0] for sub in subs]+[0] for m in uncertainty_metrics }

    # human-readable call
    '''
    sparse_curve =  {"rmse":[compute_errors_for_auc(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                     "a1":[compute_errors_for_auc(gt,pred,metrics=["a1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                     "abs":[compute_errors_for_auc(gt,pred,metrics=["abs"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
    '''
    
    # get percentiles for optimal sampling and corresponding subsets
    opt_thresholds = {m:[np.percentile(true_uncert[m], q) for q in quants] for m in uncertainty_metrics}
    opt_subs = {m:[(true_uncert[m] >= o) for o in opt_thresholds[m]] for m in uncertainty_metrics}

    # compute sparsification curves for optimal sampling (add 0 for final sampling)
    opt_curve = {m:[compute_errors_for_auc(gt,pred,metrics=[m],mask=opt_sub,reduce_mean=True)[0] for opt_sub in opt_subs[m]]+[0] for m in uncertainty_metrics}

    # compute metrics for random sampling (equal for each sampling)
    rnd_curve = {m:[compute_errors_for_auc(gt,pred,metrics=[m],mask=None,reduce_mean=True)[0] for t in range(intervals+1)] for m in uncertainty_metrics}    

    # compute error and gain metrics
    for m in uncertainty_metrics:

        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = np.trapz(sparse_curve[m], x=plotx) - np.trapz(opt_curve[m], x=plotx)
        
        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        AURG[m] = rnd_curve[m][0] - np.trapz(sparse_curve[m], x=plotx)

    # returns a dictionary with AUSE and AURG for each metric
    return {m:[AUSE[m], AURG[m]] for m in uncertainty_metrics}
