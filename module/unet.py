""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv1d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class InceptionResNetB(nn.Module):

    #"""Figure 17. The schema for 17 × 17 grid (Inception-ResNet-B) module of 
    #the Inception-ResNet-v2 network."""
    def __init__(self, input_channels, out_channels):

        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv1d(input_channels, 128, kernel_size=1),
            BasicConv1d(128, 160, kernel_size=7, padding=3),
            BasicConv1d(160, 192, kernel_size=7, padding=3)
        )

        self.branch1x1 = BasicConv1d(input_channels, 192, kernel_size=1)

        self.reduction1x1 = nn.Conv1d(384, out_channels, kernel_size=1)
        self.shortcut = nn.Conv1d(input_channels, out_channels, kernel_size=1)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch7x7(x)
        ]

        residual = torch.cat(residual, 1)

        #"""In general we picked some scaling factors between 0.1 and 0.3 to scale the residuals 
        #before their being added to the accumulated layer activations (cf. Figure 20)."""
        residual = self.reduction1x1(residual) * 0.1

        shortcut = self.shortcut(x)

        output = self.bn(residual + shortcut)
        output = self.relu(output)

        return output

class InceptionResNetA(nn.Module):

    #"""Figure 16. The schema for 35 × 35 grid (Inception-ResNet-A) 
    #module of the Inception-ResNet-v2 network."""
    def __init__(self, input_channels, out_channels):

        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv1d(input_channels, 32, kernel_size=1),
            BasicConv1d(32, 48, kernel_size=3, padding=1),
            BasicConv1d(48, 64, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv1d(input_channels, 32, kernel_size=1),
            BasicConv1d(32, 32, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv1d(input_channels, 32, kernel_size=1)

        self.reduction1x1 = nn.Conv1d(128, out_channels, kernel_size=1)
        self.shortcut = nn.Conv1d(input_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = [
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch3x3stack(x)
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)

        output = self.bn(shortcut + residual)
        output = self.relu(output)

        return output


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
            #nn.Tanh(),
        )
        # self.double_conv = nn.Sequential(InceptionResNetA(in_channels, out_channels),
        #                                 InceptionResNetA(out_channels, out_channels),
        #                                 InceptionResNetA(out_channels, out_channels))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]


        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        # oif you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.down4 = Down(256, 256)
        self.down5 = Down(256, 256)
        self.down6 = Down(256, 256)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(512, 256, bilinear)
        self.up5 = Up(512, 128, bilinear)
        self.up6 = Up(256, 64, bilinear)
        self.up7 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up3(x6, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        x = self.outc(x)
        return x
        if self.n_classes > 1:
           return F.softmax(x, dim=1)
        else:
           return torch.sigmoid(x)

if __name__ == "__main__":
    import numpy as np
    data = np.random.uniform(0, 1, (2, 1, 500))
    data = torch.from_numpy(data).float()
    net = UNet(n_channels=1, n_classes=1)
    out = net(data)
    
    print(net)
    print(out.shape)
