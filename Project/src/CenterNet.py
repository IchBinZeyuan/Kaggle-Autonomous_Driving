import numpy as np  # linear algebra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, in_ch_2, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = double_conv(in_ch + in_ch_2, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class MyUNet(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes, settings):
        super(MyUNet, self).__init__()
        self._settings = settings
        self.device = self._settings.device
        #self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model = models.resnext101_32x8d(pretrained=self._settings.pre_trained)

        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        # self.up1 = up(1282, 1024, 512, bilinear=False)
        # self.up1 = up(2050 + 1024, 512, bilinear=False)
        self.up1 = up(2050, 1024, 512, bilinear=False)
        self.up2 = up(512, 512, 256, bilinear=False)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = self.get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :, :, self._settings.img_width // 8: -self._settings.img_width // 8]
        x_center_size = x_center.size()
        base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        for param in base_model:
            param.requires_grad = not self._settings.pre_trained
        feats = base_model(x_center)
        # feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(self.device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        mesh2 = self.get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x

    def get_mesh(self, batch_size, shape_x, shape_y):
        mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
        mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
        mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
        mesh = torch.cat([torch.tensor(mg_x).to(self.device), torch.tensor(mg_y).to(self.device)], 1)
        return mesh