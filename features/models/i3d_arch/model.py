import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3d(nn.Conv3d):

    def _compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - s % self.stride[dim], 0)

    def _same_pad(self, x):
        t, h, w = x.shape[-3:]
        pad_t = self._compute_pad(0, t)
        pad_h = self._compute_pad(1, h)
        pad_w = self._compute_pad(2, w)
        pad = ( # same as TensorFlow, pad more on the right if needed
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_t // 2, pad_t - pad_t // 2
        )
        x = F.pad(x, pad)
        return x

    def forward(self, x):
        x = self._same_pad(x)
        x = super(Conv3d, self).forward(x)
        return x


class MaxPool3d(nn.MaxPool3d):

    def _compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - s % self.stride[dim], 0)

    def _same_pad(self, x):
        t, h, w = x.shape[-3:]
        pad_t = self._compute_pad(0, t)
        pad_h = self._compute_pad(1, h)
        pad_w = self._compute_pad(2, w)
        pad = ( # same as TensorFlow, pad more on the right if needed
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_t // 2, pad_t - pad_t // 2
        )
        x = F.pad(x, pad)
        return x

    def forward(self, x):
        x = self._same_pad(x)
        x = super(MaxPool3d, self).forward(x)
        return x


class Conv3dBlock(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=(1, 1, 1), 
        stride=(1, 1, 1), 
        padding=0
    ):
        super(Conv3dBlock, self).__init__()

        self.conv3d = Conv3d(
            in_channels, out_channels, kernel_size, stride, 
            padding=0, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.bn(self.conv3d(x))
        x = F.relu(x, inplace=True)
        return x


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        
        assert isinstance(out_channels, (list, tuple))
        assert len(out_channels) == 6
        for c in out_channels:
            assert isinstance(c, int)

        self.b0 = Conv3dBlock(in_channels, out_channels[0])
        self.b1a = Conv3dBlock(in_channels, out_channels[1])
        self.b1b = Conv3dBlock(out_channels[1], out_channels[2], 3)
        self.b2a = Conv3dBlock(in_channels, out_channels[3])
        self.b2b = Conv3dBlock(out_channels[3], out_channels[4], 3)
        self.b3a = MaxPool3d((3, 3, 3), (1, 1, 1))
        self.b3b = Conv3dBlock(in_channels, out_channels[5])

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        x = torch.cat([b0, b1, b2, b3], 1)
        return x


class I3D(nn.Module):

    ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c'
    )

    def __init__(self, n_classes=400, in_channels=3):
        super(I3D, self).__init__()

        self.n_classes = n_classes
        self.end_points = dict()

        # stage 1
        self.end_points['Conv3d_1a_7x7'] = Conv3dBlock(
            in_channels, 64, kernel_size=7, stride=2, padding=3
        )

        # stage 2: no temporal stride
        self.end_points['MaxPool3d_2a_3x3'] = MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2)
        )
        self.end_points['Conv3d_2b_1x1'] = Conv3dBlock(
            64, 64, kernel_size=1
        )
        self.end_points['Conv3d_2c_3x3'] = Conv3dBlock(
            64, 192, kernel_size=3, padding=1
        )

        # stage 3: no temporal stride
        self.end_points['MaxPool3d_3a_3x3'] = MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2)
        )
        self.end_points['Mixed_3b'] = InceptionModule(
            192, (64, 96, 128, 16, 32, 32)
        )
        self.end_points['Mixed_3c'] = InceptionModule(
            256, (128, 128, 192, 32, 96, 64)
        )

        # stage 4
        self.end_points['MaxPool3d_4a_3x3'] = MaxPool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2)
        )
        self.end_points['Mixed_4b'] = InceptionModule(
            128 + 192 + 96 + 64, (192, 96, 208, 16, 48, 64)
        )
        self.end_points['Mixed_4c'] = InceptionModule(
            192 + 208 + 48 + 64, (160, 112, 224, 24, 64, 64)
        )
        self.end_points['Mixed_4d'] = InceptionModule(
            160 + 224 + 64 + 64, (128, 128, 256, 24, 64, 64)
        )
        self.end_points['Mixed_4e'] = InceptionModule(
            128 + 256 + 64 + 64, (112, 144, 288, 32, 64, 64)
        )
        self.end_points['Mixed_4f'] = InceptionModule(
            112 + 288 + 64 + 64, (256, 160, 320, 32, 128, 128)
        )

        # stage 5
        self.end_points['MaxPool3d_5a_2x2'] = MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.end_points['Mixed_5b'] = InceptionModule(
            256 + 320 + 128 + 128, (256, 160, 320, 32, 128, 128)
        )
        self.end_points['Mixed_5c'] = InceptionModule(
            256 + 320 + 128 + 128, (384, 192, 384, 48, 128, 128)
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.logits = nn.Linear(384 + 384 + 128 + 128, self.n_classes)

        self._build()

    def _build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        # input size: (3, 16, 224, 224)
        for k in self.ENDPOINTS:
            x = self._modules[k](x)
        x = self.avg_pool(x).flatten(1)
        x = self.logits(x)
        return x