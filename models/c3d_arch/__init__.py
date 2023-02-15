import os

import torch
import numpy as np

from .model import C3D

root = os.path.dirname(__file__)

def build_c3d(pretrained='sports1m'):
    
    model = C3D()
    path = os.path.join(root, 'assets/{:s}.pth'.format(pretrained))
    checkpoints = torch.load(path)
    model.load_state_dict(checkpoints)

    return model

def load_mean(pretrained='sports1m'):
    
    path = os.path.join(root, 'assets/mean.npy'.format(pretrained))
    mean = np.load(path) 
    mean = torch.from_numpy(mean.astype(np.float32)).cuda()
    mean = mean[..., 8:120, 30:142]   
    
    return mean
    