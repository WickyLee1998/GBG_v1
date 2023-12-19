import torch.nn as nn
from models import resnet_cifar
from models.resnet_cifar import NormedLinear


def ResNet32Model(num_classes, use_norm=True,classifier = True):  # From LDAM_DRW
    model = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, classifier = classifier,
                                  use_norm=use_norm)
    return model
