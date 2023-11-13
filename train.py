# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch.multiprocessing as mp

from trainer import Trainer
from trainer_self import Trainer_Self
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    print(opts)
    if opts.self:        
        trainer = Trainer_Self(opts)
    else:
        trainer = Trainer(opts)

    trainer.train()
