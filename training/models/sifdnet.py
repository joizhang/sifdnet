import torch
import torch.nn.functional as F
from timm.models.layers import create_classifier
from torch import nn as nn

from training.models.aspp import DCM
from training.models.efficientnet import tf_efficientnet_b4_ns
from training.models.efficientnet_blocks import ConvBnAct
from training.models.gbb import GradientBoostNet
from training.models.resnet import Bottleneck


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


class ALAM(nn.Module):
    """
    Adjacent Level Aggregation Module (ALAM)
    """

    def __init__(self, in_channels, out_channels, size):
        super(ALAM, self).__init__()
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


class Decoder(nn.Module):

    def __init__(self, encoder_num_chs, out_channels):
        """
        hi_channels=320, mid_channels=112, lo_channels=48
        for example outputs of efficientnet-b4 are [24, 32, 56, 160, 448]
        :param encoder_num_chs:
        :param out_channels:
        """
        super(Decoder, self).__init__()
        # aspp projector
        self.dcm_proj = ConvBnAct(encoder_num_chs[4], encoder_num_chs[3], kernel_size=1, pad_type='same')
        # high level feature
        self.block1 = ConvBnAct(encoder_num_chs[3] * 2, encoder_num_chs[2], kernel_size=1, pad_type='same')
        # middle level feature
        self.block2 = ConvBnAct(encoder_num_chs[2] * 2, encoder_num_chs[0], kernel_size=1, pad_type='same')
        # low level feature
        self.block3 = ConvBnAct(encoder_num_chs[0] * 2, encoder_num_chs[0], kernel_size=1, pad_type='same')
        # output
        self.block4 = nn.Sequential(
            ConvBnAct(encoder_num_chs[0], encoder_num_chs[0], kernel_size=3, pad_type='same'),
            nn.Dropout(0.2),
            ConvBnAct(encoder_num_chs[0], out_channels, kernel_size=1, pad_type='same'),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self._initialize_weights()

    def forward(self, dcm_feat, high_level_feat, middle_level_feat, low_level_feat):
        dcm_feat = self.dcm_proj(dcm_feat)

        dcm_feat = _upsample_like(dcm_feat, high_level_feat)  # (160, 16, 16)
        high_level_feat = torch.cat((high_level_feat, dcm_feat), dim=1)  # (320, 16, 16)

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


class SIFDNet(nn.Module):

    def __init__(self, num_classes, map_classes, drop_rate, pretrained):
        super(SIFDNet, self).__init__()

        self.num_classes = num_classes
        self.map_classes = map_classes
        self.drop_rate = drop_rate

        self.encoder = tf_efficientnet_b4_ns(pretrained=pretrained, features_only=True)
        self.num_chs = [info['num_chs'] for info in self.encoder.feature_info]  # [24, 32, 56, 160, 448]
        assert len(self.num_chs) == 5
        self.alam1 = ALAM(in_channels=self.num_chs[4], out_channels=self.num_chs[3], size=(16, 16))
        self.alam2 = ALAM(in_channels=self.num_chs[2], out_channels=self.num_chs[2], size=(32, 32))
        self.alam3 = ALAM(in_channels=self.num_chs[1], out_channels=self.num_chs[0], size=(128, 128))
        self.sobel_stream = GradientBoostNet(self.num_chs, filter_type='sobel')
        self.dcm = DCM(in_channels=self.num_chs[4], atrous_rates=[1, 3, 6, 9], out_channels=self.num_chs[4])
        self.decoder = Decoder(encoder_num_chs=self.num_chs, out_channels=map_classes)
        self.global_pool, self.classifier = create_classifier(self.num_chs[4], self.num_classes, pool_type='avg')

    def forward(self, inputs):
        features = self.encoder(inputs)
        features[3] = self.alam1(features[3], features[4])
        features[2] = self.alam2(features[2], features[2])
        features[0] = self.alam3(features[0], features[1])
        x_sobel = self.sobel_stream(features[0], features[2], features[3])
        x_sobel += features[4]
        dcm_feat = self.dcm(x_sobel)
        # classification
        x = self.global_pool(dcm_feat)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        # mask
        masks_pred = self.decoder(dcm_feat, features[3], features[2], features[0])

        return x, masks_pred


def sifdnet(num_classes=2, map_classes=1, drop_rate=0., pretrained=False):
    """ GBB + ALAM + DCM """
    model = SIFDNet(num_classes=num_classes, map_classes=map_classes, drop_rate=drop_rate, pretrained=pretrained)
    model.default_cfg = model.encoder.default_cfg
    return model
