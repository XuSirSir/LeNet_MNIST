import torch
from torch import nn


# LeNet模型原本输入的图片是32x32的，但是MNIST数据集中的图片是28x28
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 使用Sequential来编写模型训练过程缺点是模型中间出了问题不好Debug
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),  # MNIST数据集中的图片是灰度图，通道数为1
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 16, 120),  # 因为输入图像大小是28*28，故这里经过计算后是4*4*16的
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        y = self.model(x)
        return y


# 测试模型
if __name__ == '__main__':
    test_input = torch.rand([1, 1, 28, 28])
    model = LeNet()
    print(model)
    test_output = model(test_input)
    print(test_output)
