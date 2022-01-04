import os
import unittest

import torch
from PIL import Image
from torch import hub
from torch import nn
from torch.backends import cudnn
from torch.utils.data import dataset
from torchsummary import summary
from torchvision import transforms, datasets

from config import Config
from training.models import tf_efficientnet_b3_ns, tf_efficientnet_b4_ns
from training.models.efficientnet import tf_efficientnet_b0_ns
from training.tools.model_utils import validate

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']

torch.backends.cudnn.benchmark = True


class EfficientNetTestCase(unittest.TestCase):

    def test_summary_tf_efficientnet_b0_ns(self):
        self.assertTrue(torch.cuda.is_available())
        model = tf_efficientnet_b0_ns(pretrained=True, num_classes=1000, in_chans=3)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)

    def test_summary_tf_efficientnet_b3_ns(self):
        self.assertTrue(torch.cuda.is_available())
        model = tf_efficientnet_b3_ns(pretrained=True, num_classes=1000, in_chans=3, features_only=True)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)

    def test_tf_efficientnet_b3_ns(self):
        model = tf_efficientnet_b3_ns(pretrained=True, num_classes=1000, in_chans=3)
        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        valdir = os.path.join(CONFIG['IMAGENET_HOME'], 'val')
        self.assertEqual(True, os.path.exists(valdir))

        input_size = model.default_cfg['input_size']
        resize = int(input_size[1] / model.default_cfg['crop_pct'])
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(resize, Image.BICUBIC),
                transforms.CenterCrop((input_size[1], input_size[2])),
                transforms.ToTensor(),
                transforms.Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std']),
            ])),
            batch_size=20, shuffle=False,
            num_workers=1, pin_memory=True)

        validate(val_loader, model, criterion)

    def test_summary_tf_efficientnet_b4_ns(self):
        self.assertTrue(torch.cuda.is_available())
        model = tf_efficientnet_b4_ns(pretrained=True, num_classes=1000, in_chans=3)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)


if __name__ == '__main__':
    unittest.main()
