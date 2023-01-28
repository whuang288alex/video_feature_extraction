import os

import torch

from .model import I3D

root = os.path.dirname(__file__)
I3D_PATH = {
    'rgb_kinetics': os.path.join(root, 'assets/rgb_kinetics.pth'),
    'rgb_charades': os.path.join(root, 'assets/rgb_charades.pth'),
    'flow_kinetics': os.path.join(root, 'assets/flow_kinetics.pth'),
    'flow_charades': os.path.join(root, 'assets/flow_charades.pth'),
}

def build_i3d(pretrained='rgb_kinetics'):
    if pretrained == 'rgb_kinetics':
        net = I3D(400, 3)
    elif pretrained == 'rgb_charades':
        net = I3D(157, 3)
    elif pretrained == 'flow_kinetics':
        net = I3D(400, 2)
    elif pretrained == 'flow_charades':
        net = I3D(157, 2)
    else:
        raise NotImplementedError('pretrained model does not exist')
    
    ckpt = torch.load(
        I3D_PATH[pretrained], map_location=lambda storage, loc: storage
    )
    net.load_state_dict(ckpt)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    return net