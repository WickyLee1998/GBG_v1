'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torch.distributions import normal

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features, set_s = True):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.set_s = set_s

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        if self.set_s:
            s = 30
            return s*cosine
        else:
            return cosine

class GCl_NormLinear(nn.Module):
    def __init__(self, in_features, out_features, cls_num_list, set_s = False):
        super(GCl_NormLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list
        self.m = 0 # same as GCL
        self.s = 30 # same as GCL
        self.noise_mul = 0.5 # same as GCL 0.5
        self.simpler = normal.Normal(0, 1/3)
        #self.set_s = set_s

    def forward(self, x, target=None):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        if target!=None:
            index = torch.zeros_like(cosine, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device)

        if self.m_list.max() == 0:
            cosine = cosine
        else:
            cosine = cosine - self.noise_mul * noise.abs() / self.m_list.max() * self.m_list
        if target!=None:
            output = torch.where(index, cosine - self.m, cosine)
            return self.s * output
        #out = x.mm(self.weight)

        return self.s*cosine

class NoiseLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NoiseLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.simpler = normal.Normal(0, 1/10000000)
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        noise  = torch.randn_like(input) #noise = self.simpler.sample(input.shape).clamp(-1, 1).to(input.device)#
        
        out = F.linear(input, self.weight, self.bias)
        noise = F.linear(noise, self.weight)
        output = [out,noise]
        return output

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)      
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10,
                 classifier = True, use_norm= False, use_noise = False, set_s = False, use_myGCL = False,
                 cls_num_list=None):
        super(ResNet_s, self).__init__()
        if cls_num_list is None:
            cls_num_list = [0, 0]
        self.in_planes = 16
        self.set_s = set_s
        self.classifier = classifier
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    
        if self.classifier:
            if use_norm:
                self.fc = NormedLinear(64, num_classes, set_s = self.set_s)
            elif use_noise:
                self.fc = NoiseLinear(64, num_classes)
            elif use_myGCL:
                self.fc = GCl_NormLinear(64, num_classes, cls_num_list=cls_num_list)
            else:
                self.fc = nn.Linear(64, num_classes)
        feat_dim = 32
        self.head = nn.Sequential(nn.Linear(64, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                  nn.Linear(512, 128))
        self.head_center = nn.Sequential(nn.Linear(64, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                     nn.Linear(512, 128))
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, get_feat=False, target = None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feat = F.avg_pool2d(out, out.size()[3])
        feat = feat.view(feat.size(0), -1)
        feat_mlp = F.normalize(self.head(feat), dim=1)
        temp = self.head_center(self.fc.weight.T)
        centers_logits = F.normalize(temp, dim=1)
        if self.classifier:
            if target!=None:
                logits = self.fc(feat,target)
            else:
                logits = self.fc(feat)

            if get_feat == True:
                out = dict()  
                out['feature'] = feat
                out['score'] = logits
            else:
                out = logits
            return feat_mlp, out, centers_logits
        else:
            return feat
            

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, classifier=True, use_norm = False, use_noise=False,set_s = False,use_myGCL=False,cls_num_list = None):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, 
                    classifier = classifier, use_norm= use_norm, use_noise=use_noise ,set_s=set_s,use_myGCL=use_myGCL,
                    cls_num_list = cls_num_list)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()