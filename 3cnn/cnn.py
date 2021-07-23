import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
#import torch.nn.functional as F
import numpy as np

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib
import matplotlib.pyplot as plt

#参数表

kernel_size_cov=3           #卷积核大小
out__chan1=5               #第一卷积层输出通道数
out_chan2=10                #第二卷积层输出通道数
kernel_size_pol=2          #池化层核大小
features=1200               #第一连接层输出数量
learning_rate = 0.001      #学习率
batch_size = 100            #批处理量
epochs=1                #迭代次数




class cnn_mnist(nn.Module):                  #CNN继承了nn.Module 类
    def __init__(self):
        super(cnn_mnist, self).__init__()
        '''
        卷积层
        池化层  
        交替各两层
        完全连接层两层
        '''
        #选用ReLU激活函数
        self.act=nn.ReLU()
        #MNIST数据通道数为1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out__chan1, kernel_size=kernel_size_cov, stride=1, padding=1)   
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size_pol, stride=2)

        self.conv2 = nn.Conv2d(in_channels=out__chan1, out_channels=out_chan2, kernel_size=kernel_size_cov, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=kernel_size_pol,stride=2)

        self.fc1 = nn.Linear(in_features=49*out_chan2, out_features=features)
        self.fc2 = nn.Linear(in_features=features, out_features=10)
    
    def forward(self, input):
        #向前传播 output:x
        x = self.conv1(input)
        x = self.act(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)

        x = x.view(-1, 49*out_chan2)
        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)
        return x


def showimg(img,title=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# main

# 载入MNIST训练集
print("=====MNIST data loading=====")
train_dataset = MNIST(root='.',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset, shuffle=True,batch_size=batch_size) 
print("=====sucessfully download!=======")
#训练数据放入迭代器
dataiter = iter(train_loader)
batch = next(dataiter)

#showimg(make_grid(batch[0],nrow=10,padding=2,pad_value=1),'训练集部分数据')   展示部分数据集合

#初始化
cnn=cnn_mnist()
criterion = nn.CrossEntropyLoss()    #交叉熵损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate,weight_decay=0.001)  #优化器

Loss_list = []     
train_total=0
print("=======start training!=======")
for epoch in range(epochs):
    for i, (imgdata, label) in enumerate(train_loader):
            train_total+=1
            optimizer.zero_grad()
            output = cnn(imgdata)
            loss = criterion(output, label)
            loss.backward()          #逆传播
            Loss_list.append(loss.data)
            optimizer.step()
    print("Training Epoch {} Completed".format(epoch))
#获取测试集
test_dataset = MNIST(root='.',train=False,transform=transforms.ToTensor(),download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=batch_size)   
#测试数据放入迭代器
dataiter = iter(test_loader)
batch = next(dataiter)   

total = 0
test_total=0
correct = 0
Val_loss_list=[]
for i, (test_img, test_label) in enumerate(test_loader):
    # 正向通过神经网络得到预测结果
    outputs = cnn(test_img)
    _,predicted = torch.max(outputs, 1)
    test_total+=1
    val_loss= criterion(outputs, test_label)
    Val_loss_list.append(val_loss.data) 
        # 总数和正确数 
    total += len(test_label)
    correct += (predicted == test_label).sum()


torch.save(cnn, 'cnn_minist.pkl')


accuracy = correct.item()/ total
print("total is", total)
print("correct is", correct.item())
print('Testing Results:\n  Loss: {}  \nAccuracy: {} %'.format(loss.data, accuracy*100))

print("validation loss is",val_loss.item())

loss_x=range(train_total)
plt.plot(loss_x,Loss_list, 'o-')
plt.title('Loss Curve')
plt.ylabel('train loss')
plt.xlabel('iters')
plt.show()

    