import torch
import torch.nn as nn


kernel_size_cov=3           #卷积核大小
out__chan1=5               #第一卷积层输出通道数
out_chan2=10                #第二卷积层输出通道数
kernel_size_pol=2          #池化层核大小
features=1200               #第一连接层输出数量
learning_rate = 0.001      #学习率
batch_size = 100            #批处理量
epochs=10                #迭代次数
class cnn_mnist(nn.Module):  # CNN继承了nn.Module 类
    def __init__(self):
        super(cnn_mnist, self).__init__()
        '''
        卷积层
        池化层  
        交替各两层
        完全连接层两层
        '''
        # 选用ReLU激活函数
        self.act = nn.ReLU()
        # MNIST数据通道数为1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out__chan1, kernel_size=kernel_size_cov, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size_pol, stride=2)

        self.conv2 = nn.Conv2d(in_channels=out__chan1, out_channels=out_chan2, kernel_size=kernel_size_cov, stride=1,
                               padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=kernel_size_pol, stride=2)

        self.fc1 = nn.Linear(in_features=49 * out_chan2, out_features=features)
        self.fc2 = nn.Linear(in_features=features, out_features=10)

    def forward(self, input):
        # 向前传播 output:x
        x = self.conv1(input)
        x = self.act(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)

        x = x.view(-1, 49 * out_chan2)
        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)
        return x

cnn=torch.load('cnn_minist.pkl', map_location='cpu')    #读取cnn
