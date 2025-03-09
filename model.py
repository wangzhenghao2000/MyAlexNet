import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=400,out_features=120)
        self.f6 = nn.Linear(in_features=120,out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self,x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

import torch.nn.functional as F
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.s2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.c3 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.c5 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.c6 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1)
        self.c7 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(6*6*256,4096)
        self.f2 = nn.Linear(4096,4096)
        self.f3 = nn.Linear(4096,10)
    def forward(self,x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))
        x = self.ReLU(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)
        x = self.ReLU(self.f1(x))
        x = F.dropout(x,0.5)
        x = self.ReLU(self.f2(x))
        x = F.dropout(x,0.5)
        x = self.f3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = LeNet().to(device)
    # print(summary(model,(1,28,28)))
    model = AlexNet().to(device)
    print(summary(model,(1,227,227)))