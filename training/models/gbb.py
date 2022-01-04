import torch
from torch import nn as nn

from timm.models.layers import create_conv2d

__all__ = ['GradientBoostNet']

SOBEL_X = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]
SOBEL_Y = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
SCHARR_X = [[3., 0., -3.], [10., 0., -10.], [3., 0., -3.]]
SCHARR_Y = [[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]]


class GradientBoostBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pwl_stride, pwl_padding, filter_type='sobel'):
        super(GradientBoostBlock, self).__init__()
        self.in_channels = in_channels

        if filter_type == 'sobel':
            filter_x = torch.tensor(SOBEL_X)
            filter_y = torch.tensor(SOBEL_Y)
        else:
            filter_x = torch.tensor(SCHARR_X)
            filter_y = torch.tensor(SCHARR_Y)
        self.conv_grad_x = create_conv2d(in_channels, in_channels, kernel_size=3, stride=1, depthwise=True)
        self.conv_grad_x.weight = nn.Parameter(filter_x.repeat(in_channels, 1, 1, 1))
        self.conv_grad_y = create_conv2d(in_channels, in_channels, kernel_size=3, stride=1, depthwise=True)
        self.conv_grad_y.weight = nn.Parameter(filter_y.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.bn_grad = nn.BatchNorm2d(in_channels)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(in_channels, in_channels, kernel_size=3, stride=2, depthwise=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        # Point-wise linear projection
        self.conv_pw = create_conv2d(in_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.ReLU(inplace=True)
        # Point-wise linear projection
        self.conv_pwl = create_conv2d(in_channels, out_channels, kernel_size=3, stride=pwl_stride, padding=pwl_padding)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Depth-wise expansion
        out = self.conv_dw(x)
        out = self.bn1(out)
        out = self.act1(out)

        # Gradient boost
        with torch.no_grad():
            grad_x = self.conv_grad_x(out)
            grad_y = self.conv_grad_y(out)
            grad = torch.sqrt(torch.square(grad_x) + torch.square(grad_y))
        grad = self.bn_grad(grad)
        out = out + grad

        # Point-wise convolution
        out = self.conv_pw(out)
        out = self.bn2(out)
        out = self.act2(out)

        # Point-wise linear projection
        out = self.conv_pwl(out)
        out = self.bn3(out)
        return out


class GradientBoostNet(nn.Module):

    def __init__(self, num_chs, filter_type):
        super(GradientBoostNet, self).__init__()
        self.block1 = GradientBoostBlock(num_chs[0], num_chs[2], pwl_stride=2, pwl_padding=1, filter_type=filter_type)
        self.block2 = GradientBoostBlock(num_chs[2], num_chs[3], pwl_stride=1, pwl_padding=1, filter_type=filter_type)
        self.block3 = GradientBoostBlock(num_chs[3], num_chs[4], pwl_stride=1, pwl_padding=1, filter_type=filter_type)

    def forward(self, feat_0, feat_2, feat_3):
        x = self.block1(feat_0) + feat_2
        x = self.block2(x) + feat_3
        x = self.block3(x)
        return x
