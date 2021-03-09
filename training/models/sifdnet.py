import torch
import torch.nn.functional as F
from timm.models.efficientnet_blocks import ConvBnAct
from timm.models.layers import create_classifier
from torch import nn as nn

from training.models.aspp import ASPP
from training.models.efficientnet import tf_efficientnet_b4_ns
from training.models.resnet import Bottleneck

__all__ = ['sifdnet', 'BottomUpTopDownAttention', 'Decoder', 'Alam']


def _upsample_like(src, tgt):
    src = F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=True)
    return src


class BottomUpTopDownAttention(nn.Module):

    def __init__(self, in_channels, size: tuple):
        super(BottomUpTopDownAttention, self).__init__()
        # Bottom up top down
        # self.trunk_block = Bottleneck(inplanes=in_channels, planes=in_channels // 4)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            # Bottleneck has expansion coefficient 4, so in_channels divided by 4
            Bottleneck(inplanes=in_channels, planes=in_channels // 4),
            Bottleneck(inplanes=in_channels, planes=in_channels // 4),
        )
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size)
        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # mask branch
        mask = self.max_pool1(x)
        mask = self.softmax1_blocks(mask)
        mask = self.interpolation1(mask)
        mask = self.softmax2_blocks(mask)

        # trunk branch
        # x = self.trunk_block(x)
        x = (1 + mask) * x
        return x


class Decoder(nn.Module):

    def __init__(self, hi_channels=320, mid_channels=112, lo_channels=48, out_channels=1):
        super(Decoder, self).__init__()

        # high level feature
        self.block1 = ConvBnAct(hi_channels, mid_channels // 2, kernel_size=1, pad_type='same')
        # middle level feature
        self.block2 = ConvBnAct(mid_channels, lo_channels // 2, kernel_size=1, pad_type='same')
        # low level feature
        self.block3 = ConvBnAct(lo_channels, 24, kernel_size=1, pad_type='same')
        # output
        self.block4 = nn.Sequential(
            ConvBnAct(24, 24, kernel_size=3, pad_type='same'),
            nn.Dropout(0.2),
            ConvBnAct(24, out_channels, kernel_size=1, pad_type='same'),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self._initialize_weights()

    def forward(self, aspp_feat, high_level_feat, middle_level_feat, low_level_feat):
        """
        :param aspp_feat: (160, 8, 8)
        :param high_level_feat: (160, 16, 16)
        :param middle_level_feat: (56, 32, 32)
        :param low_level_feat: (24, 128, 128)
        :return:
        """
        aspp_feat = _upsample_like(aspp_feat, high_level_feat)  # (160, 16, 16)
        high_level_feat = torch.cat((high_level_feat, aspp_feat), dim=1)  # (320, 16, 16)

        high_level_feat = self.block1(high_level_feat)  # (56, 16, 16)
        high_level_feat = _upsample_like(high_level_feat, middle_level_feat)  # (56, 32, 32)
        middle_level_feat = torch.cat((middle_level_feat, high_level_feat), dim=1)  # (112, 32, 32)

        middle_level_feat = self.block2(middle_level_feat)  # (24, 32, 32)
        middle_level_feat = _upsample_like(middle_level_feat, low_level_feat)  # (24, 128, 128)
        low_level_feat = torch.cat((low_level_feat, middle_level_feat), dim=1)  # (48, 128, 128)

        low_level_feat = self.block3(low_level_feat)  # (24, 128, 128)
        low_level_feat = self.block4(low_level_feat)  # (1, 128, 128)
        low_level_feat = self.up(low_level_feat)  # (1, 256, 256)

        return low_level_feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Alam(nn.Module):
    """
    Adjacent Level Aggregation Module (ALAM)
    """

    def __init__(self, in_channels, out_channels, size):
        super(Alam, self).__init__()
        self.block1 = ConvBnAct(in_channels, out_channels, kernel_size=3, dilation=2, pad_type='same')
        self.block2 = ConvBnAct(out_channels * 2, out_channels, kernel_size=3, pad_type='same')
        self.attn = BottomUpTopDownAttention(out_channels, size=size)

    def forward(self, x1, x2):
        x2 = self.block1(x2)
        x2 = _upsample_like(x2, x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.block2(x1)
        x1 = self.attn(x1)
        return x1


class SIFDNet(nn.Module):

    def __init__(self, num_classes=2, map_classes=1, drop_rate=0., pretrained=False):
        super(SIFDNet, self).__init__()

        self.num_classes = num_classes
        self.map_classes = map_classes
        self.drop_rate = drop_rate

        self.encoder = tf_efficientnet_b4_ns(pretrained=pretrained, features_only=True)
        self.num_chs = [info['num_chs'] for info in self.encoder.feature_info]  # [24, 32, 56, 160, 448]
        assert len(self.num_chs) == 5
        self.alam1 = Alam(in_channels=self.num_chs[4], out_channels=self.num_chs[3], size=(16, 16))
        self.alam2 = Alam(in_channels=self.num_chs[1], out_channels=self.num_chs[0], size=(128, 128))
        self.global_pool, self.classifier = create_classifier(self.num_chs[4], self.num_classes, pool_type='avg')
        self.aspp = ASPP(in_channels=self.num_chs[4], atrous_rates=[12, 24, 36], out_channels=self.num_chs[3])
        self.decoder = Decoder(out_channels=map_classes)

    def forward(self, inputs):
        features = self.encoder(inputs)
        aspp_feat = self.aspp(features[-1])
        features[-2] = self.alam1(features[-2], features[-1])
        features[0] = self.alam2(features[0], features[1])
        # low_level_feat, middle_level_feat, high_level_feat, aspp_feat
        mask = self.decoder(aspp_feat, features[-2], features[2], features[0])
        x = self.global_pool(features[-1])
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x, mask


def sifdnet(num_classes=2, map_classes=1, pretrained=False):
    model = SIFDNet(num_classes=num_classes, map_classes=map_classes, pretrained=pretrained)
    model.default_cfg = model.encoder.default_cfg
    return model
