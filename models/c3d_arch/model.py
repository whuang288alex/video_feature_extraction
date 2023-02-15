import torch
import torch.nn as nn


class C3D(nn.Module):

    def __init__(self, n_classes=487):
        super(C3D, self).__init__()

        # (3, 16, 112, 112)
        self.conv1 = nn.Conv3d(3, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        # (64, 16, 56, 56)
        self.conv2 = nn.Conv3d(64, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool3d(2)

        # (128, 8, 28, 28)
        self.conv3a = nn.Conv3d(128, 256, 3, 1, 1)
        self.conv3b = nn.Conv3d(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool3d(2)

        # (256, 4, 14, 14)
        self.conv4a = nn.Conv3d(256, 512, 3, 1, 1)
        self.conv4b = nn.Conv3d(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool3d(2)

        # (512, 2, 7, 7)
        self.conv5a = nn.Conv3d(512, 512, 3, 1, 1)
        self.conv5b = nn.Conv3d(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool3d(2, padding=(0, 1, 1))

        # (512 * 4 * 4,)
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, n_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # input size: (3, 16, 112, 112)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.flatten(1)
        fc6 = self.relu(self.fc6(x))
        fc7 = self.relu(self.fc7(fc6))

        # out = {'fc6': fc6, 'fc7': fc7}
        return torch.cat([fc6, fc7], dim = 1)