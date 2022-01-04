import torch.nn.functional as F
from timm.models.layers import create_classifier
from torch import nn as nn

from training.models.efficientnet import tf_efficientnet_b4_ns
from training.models.efficientnet_blocks import ConvBnAct

__all__ = ['sifdnet_1']

from training.models.gbb import GradientBoostNet


class SIFDNet_1(nn.Module):

    def __init__(self, num_classes, map_classes, drop_rate, pretrained):
        super(SIFDNet_1, self).__init__()

        self.num_classes = num_classes
        self.map_classes = map_classes
        self.drop_rate = drop_rate

        self.encoder = tf_efficientnet_b4_ns(pretrained=pretrained, features_only=True)
        self.num_chs = [info['num_chs'] for info in self.encoder.feature_info]  # [24, 32, 56, 160, 448]
        assert len(self.num_chs) == 5
        self.sobel_stream = GradientBoostNet(self.num_chs, filter_type='sobel')
        self.dcm = ConvBnAct(self.num_chs[4], self.num_chs[4], kernel_size=1, pad_type='same')
        self.global_pool, self.classifier = create_classifier(self.num_chs[4], self.num_classes, pool_type='avg')

    def forward(self, inputs):
        features = self.encoder(inputs)
        x_sobel = self.sobel_stream(features[0], features[2], features[3])
        x_sobel += features[4]
        dcm_feat = self.dcm(x_sobel)

        x = self.global_pool(dcm_feat)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)

        return x


def sifdnet_1(num_classes=2, map_classes=1, drop_rate=0., pretrained=False):
    """ GBB """
    model = SIFDNet_1(num_classes=num_classes, map_classes=map_classes, drop_rate=drop_rate, pretrained=pretrained)
    model.default_cfg = model.encoder.default_cfg
    return model
