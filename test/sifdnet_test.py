import os
import unittest

import torch
from pytorch_toolbelt.utils import count_parameters
from torch import hub
from torch.backends import cudnn
from torchsummary import summary

from config import Config
from training.models import sifdnet
from training.models.aspp import ASPP
from training.models.sifdnet import BottomUpTopDownAttention, Decoder, Alam

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']

torch.backends.cudnn.benchmark = True


class SIFDNetTestCase(unittest.TestCase):

    def test_sifdnet(self):
        self.assertTrue(torch.cuda.is_available())
        model = sifdnet(pretrained=True)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)

    def test_aspp(self):
        self.assertTrue(torch.cuda.is_available())
        model = ASPP(in_channels=448, atrous_rates=[12, 24, 36], out_channels=160)
        model = model.cuda()
        input_size = (448, 8, 8)
        summary(model, input_size=input_size)

    def test_bottom_up_top_down_attention(self):
        self.assertTrue(torch.cuda.is_available())
        model = BottomUpTopDownAttention(24, (128, 128))
        model = model.cuda()
        input_size = (24, 128, 128)
        summary(model, input_size=input_size)

    def test_alam1(self):
        self.assertTrue(torch.cuda.is_available())
        model = Alam(in_channels=448, out_channels=160, size=(16, 16))
        model = model.cuda()
        input_size = [(160, 16, 16), (448, 8, 8)]
        summary(model, input_size=input_size)

    def test_alam2(self):
        self.assertTrue(torch.cuda.is_available())
        model = Alam(in_channels=32, out_channels=24, size=(128, 128))
        model = model.cuda()
        input_size = [(24, 128, 128), (32, 64, 64)]
        summary(model, input_size=input_size)

    def test_decoder(self):
        self.assertTrue(torch.cuda.is_available())
        model = Decoder()
        model = model.cuda()
        input_size = [(24, 128, 128), (56, 32, 32), (160, 16, 16), (160, 8, 8)]
        summary(model, input_size=input_size)

    def test_count_sifdnet_parameters(self):
        self.assertTrue(torch.cuda.is_available())
        model = sifdnet(pretrained=True)
        print(count_parameters(model))


if __name__ == '__main__':
    unittest.main()
