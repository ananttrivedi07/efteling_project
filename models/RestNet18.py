from torch import nn
from PIL import Image
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.short = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.short = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.short(x)
        return F.relu(y)

class ResNet18(nn.Module):
    def __init__(self, num_classes=6, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 7, 2, 3, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.pool  = nn.MaxPool2d(3,2,1)
        self.layer1 = self._make_layer(width,       width, 2, 1)
        self.layer2 = self._make_layer(width,     width*2, 2, 2)
        self.layer3 = self._make_layer(width*2,   width*4, 2, 2)
        self.layer4 = self._make_layer(width*4,   width*4, 2, 2)
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(width*4, num_classes))
        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
    def _make_layer(self, in_p, planes, blocks, stride):
        layers = [BasicBlock(in_p, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)