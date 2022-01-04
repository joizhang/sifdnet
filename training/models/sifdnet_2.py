import torch.nn.functional as F
from timm.models.layers import create_classifier
from torch import nn as nn

from training.models.efficientnet import tf_efficientnet_b4_ns
from training.models.efficientnet_blocks import ConvBnAct
from training.models.sifdnet import ALAM, Decoder

__all__ = ['sifdnet_2']


class SIFDNet_2(nn.Module):

    def __init__(self, num_classes, map_classes, drop_rate, pretrained):
        super(SIFDNet_2, self).__init__()

        self.num_classes = num_classes
        self.map_classes = map_classes
        self.drop_rate = drop_rate

        self.encoder = tf_efficientnet_b4_ns(pretrained=pretrained, features_only=True)
        self.num_chs = [info['num_chs'] for info in self.encoder.feature_info]  # [24, 32, 56, 160, 448]
        assert len(self.num_chs) == 5
        self.alam1 = ALAM(in_channels=self.num_chs[4], out_channels=self.num_chs[3], size=(16, 16))
        self.alam2 = ALAM(in_channels=self.num_chs[2], out_channels=self.num_chs[2], size=(32, 32))
        self.alam3 = ALAM(in_channels=self.num_chs[1], out_channels=self.num_chs[0], size=(128, 128))
        self.dcm = ConvBnAct(self.num_chs[4], self.num_chs[4], kernel_size=1, pad_type='same')
        self.decoder = Decoder(encoder_num_chs=self.num_chs, out_channels=map_classes)
        self.global_pool, self.classifier = create_classifier(self.num_chs[4], self.num_classes, pool_type='avg')

    def forward(self, inputs):
        features = self.encoder(inputs)
        features[3] = self.alam1(features[3], features[4])
        features[2] = self.alam2(features[2], features[2])
        features[0] = self.alam3(features[0], features[1])
        dcm_feat = self.dcm(features[-1])

        x = self.global_pool(dcm_feat)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)

        masks_pred = self.decoder(dcm_feat, features[3], features[2], features[0])

        return x, masks_pred


def sifdnet_2(num_classes=2, map_classes=1, drop_rate=0., pretrained=False):
    """ ALAM """
    model = SIFDNet_2(num_classes=num_classes, map_classes=map_classes, drop_rate=drop_rate, pretrained=pretrained)
    model.default_cfg = model.encoder.default_cfg
    return model
