import os
import unittest

import torch
from torch import hub
from torch.backends import cudnn
from torchsummary import summary

from config import Config
from training.models import sifdnet, sifdnet_1, sifdnet_2, sifdnet_3
from training.models.aspp import ASPP
from training.models.sifdnet import ALAM, Decoder

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']

torch.backends.cudnn.benchmark = True


class SifdNetTestCase(unittest.TestCase):

    def test_sifdnet(self):
        self.assertTrue(torch.cuda.is_available())
        model = sifdnet(pretrained=True)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)

    def test_aspp(self):
        self.assertTrue(torch.cuda.is_available())
        model = ASPP(in_channels=448, atrous_rates=[12, 24, 36], out_channels=448)
        model = model.cuda()
        input_size = (448, 8, 8)
        summary(model, input_size=input_size)

    def test_alam1(self):
        self.assertTrue(torch.cuda.is_available())
        model = ALAM(in_channels=448, out_channels=160, size=(16, 16))
        model = model.cuda()
        input_size = [(160, 16, 16), (448, 8, 8)]
        summary(model, input_size=input_size)

    def test_alam2(self):
        self.assertTrue(torch.cuda.is_available())
        model = ALAM(in_channels=32, out_channels=24, size=(128, 128))
        model = model.cuda()
        input_size = [(24, 128, 128), (32, 64, 64)]
        summary(model, input_size=input_size)

    def test_decoder(self):
        self.assertTrue(torch.cuda.is_available())
        model = Decoder(encoder_num_chs=[24, 32, 56, 160, 448], out_channels=2)
        model = model.cuda()
        input_size = [(448, 8, 8), (160, 16, 16), (56, 32, 32), (24, 128, 128)]
        summary(model, input_size=input_size)

    def test_sifdnet_1(self):
        self.assertTrue(torch.cuda.is_available())
        model = sifdnet_1(pretrained=True)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)

    def test_sifdnet_2(self):
        self.assertTrue(torch.cuda.is_available())
        model = sifdnet_2(pretrained=True)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)

    def test_sifdnet_3(self):
        self.assertTrue(torch.cuda.is_available())
        model = sifdnet_3(pretrained=True)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)


if __name__ == '__main__':
    unittest.main()
