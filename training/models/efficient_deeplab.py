import torch
import torch.nn.functional as F
from torch import nn as nn

from training.models.aspp import ASPP
from training.models.efficientnet import tf_efficientnet_b4_ns
from training.models.efficientnet_blocks import ConvBnAct
from training.models.layers import create_classifier

__all__ = ['efficient_deeplab', 'Decoder']


class Decoder(nn.Module):

    def __init__(self, middle_level_channels=952, aspp_channels=80):
        super(Decoder, self).__init__()

        # middle level feature
        self.block1 = ConvBnAct(448, 56, kernel_size=1, pad_type='same')
        self.block2 = ConvBnAct(168, 56, kernel_size=1, pad_type='same')

        # low level feature
        self.block3 = ConvBnAct(56, 24, kernel_size=1, pad_type='same')
        self.block4 = ConvBnAct(48, 24, kernel_size=1, pad_type='same')

        # output
        self.block5 = nn.Sequential(
            ConvBnAct(24, 24, kernel_size=3, pad_type='same'),
            nn.Dropout(0.2),
            ConvBnAct(24, 1, kernel_size=1, pad_type='same'),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self._initialize_weights()

    @staticmethod
    def _upsample_like(src, tgt):
        src = F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=True)
        return src

    def forward(self, low_level_feat, middle_level_feat, high_level_feat, aspp_feat):
        high_level_feat = self.block1(high_level_feat)  # (56, 8, 8)
        high_level_feat = self._upsample_like(high_level_feat, middle_level_feat)  # (56, 32, 32)
        aspp_feat = self._upsample_like(aspp_feat, middle_level_feat)  # (56, 32, 32)
        middle_level_feat = torch.cat((middle_level_feat, high_level_feat, aspp_feat), dim=1)  # (168, 32, 32)
        middle_level_feat = self.block2(middle_level_feat)  # (56, 32, 32)

        middle_level_feat = self.block3(middle_level_feat)  # (24, 32, 32)
        middle_level_feat = self._upsample_like(middle_level_feat, low_level_feat)  # (24, 128, 128)
        low_level_feat = torch.cat((low_level_feat, middle_level_feat), dim=1)  # (48, 128, 128)
        low_level_feat = self.block4(low_level_feat)  # (24, 128, 128)

        low_level_feat = self.block5(low_level_feat)  # (1, 128, 128)
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


class EfficientDeepLab(nn.Module):

    def __init__(self, num_classes=2, num_features=448, drop_rate=0., pretrained=False):
        super(EfficientDeepLab, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.low_level_channels = 32
        self.high_level_channels = 448
        self.aspp_channels = 56

        self.encoder = tf_efficientnet_b4_ns(pretrained=pretrained, features_only=True, out_indices=[0, 2, 4])
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type='avg')
        self.aspp = ASPP(self.high_level_channels, atrous_rates=[12, 24, 36], out_channels=self.aspp_channels)
        self.decoder = Decoder(self.low_level_channels, self.aspp_channels)

    def forward(self, inputs):
        # low_level_feat, middle_level_feat, high_level_feat
        low_level_feat, middle_level_feat, high_level_feat = self.encoder(inputs)
        aspp_feat = self.aspp(high_level_feat)
        mask = self.decoder(low_level_feat, middle_level_feat, high_level_feat, aspp_feat)
        x = self.global_pool(high_level_feat)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x, mask

    def _get_params(self, modules):
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_1x_lr_params(self):
        modules = [self.encoder, self.classifier]
        yield from self._get_params(modules)

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        yield from self._get_params(modules)


def efficient_deeplab(pretrained=False):
    model = EfficientDeepLab(pretrained=pretrained)
    model.default_cfg = model.encoder.default_cfg
    return model
