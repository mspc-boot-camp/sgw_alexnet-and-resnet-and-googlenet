from torch import nn
import torch
#vgg16网络结构含有参数的网络层一共有16层，即13个卷积层，5个池化层，3个全连接层，不包括激活层。
class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=False):  # 自己是几种就把这个7改成几
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(4608, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()  # 参数初始化

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历各个层进行参数初始化
            if isinstance(m, nn.Conv2d):  # 如果是卷积层的话 进行下方初始化
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # 正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 如果偏置不是0 将偏置置成0  相当于对偏置进行初始化
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.xavier_uniform_(m.weight)  # 也进行正态分布初始化
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  # 将所有偏执置为0


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model


VGGnet = vgg(num_classes=2, init_weights=True)  # 将模型命名为#自己是几种就把这个7改成几
