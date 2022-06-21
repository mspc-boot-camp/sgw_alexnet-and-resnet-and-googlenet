import torch.nn as nn
#alexnet包括五个卷积层、3个全连接层
class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 120, 120]  output[48, 55, 55] ，此处我的计算结果和他的结果不一样
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
            nn.Flatten(),#为将连续的几个维度展平成一个tensor（将一些维度合并）
            nn.Dropout(p=0.5),#dropout通过设置好的概率随机将某个隐藏层神经元的输出设置为0，因此这个神经元将不参与前向传播和反向传播，在下一次迭代中
            # 会根据概率重新将某个神经元的输出置0。这样一来，在每次迭代中都能够尝试不同的网络结构，通过组合多个模型的方式有效地减少过拟合。
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),#这一层一半的神经元不参与前向传播和反向传播
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),  # 自己的数据集是几种，这个7就设置为几
        )
    def forward(self, x):
        x = self.model(x)
        return x