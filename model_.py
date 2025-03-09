import copy
import time
from pickletools import optimize

import torch
from torch import nn
from torch.cuda import device
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data

class LeNetP(nn.Module):
    def __init__(self):
        super(LeNetP,self).__init__()
        self.sig = nn.Sigmoid()
        self.c1  = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=400,out_features=120)
        self.f6 = nn.Linear(in_features=120,out_features=84)
        self.f7 = nn.Linear(in_features=84,out_features=10)

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

def train_val_data_process():
    train_dataset = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                              download=True
                              )
    train_data,val_data = Data.random_split(train_dataset,[round(0.8*len(train_dataset)),round(0.2*len(train_dataset))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=4)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=4)
    return train_dataloader,val_dataloader

def train_model(model,train_dataloader,val_dataloader,epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    for epoch in range(epochs):
        time_use = time.time()
        print("-"*10)
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()

            output = model(b_x)
            loss = criterion(output,b_y)
            pre_lab = torch.argmax(output,dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab==b_y.data)
            train_num += b_x.size(0)
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            loss = criterion(output,b_y)

            val_loss += loss.item()*b_x.size(0)
            val_corrects += torch.sum(pre_lab==b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)

        time_use = time.time()-time_use

        if val_acc_all[-1]>best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        print('{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))
        print('Time: {:.4f} m {:.4f} s'.format(time_use//60,time_use%60))
    import os
    folder_path = './model'
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))])+1

    torch.save(best_model_wts,f'./model/best_model{file_count}.pth')
    print("Done")
    return f'./model/best_model{file_count}.pth'

def test_data_process():
    test_dataset = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                              download=True
                              )

    test_dataloader = Data.DataLoader(dataset=test_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=6)

    return test_dataloader
def test_model(model,test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    test_correct = 0.0
    test_num = 0

    with torch.no_grad():
        for t_x,t_y in test_dataloader:
            t_x = t_x.to(device)
            t_y = t_y.to(device)
            model.eval()
            output = model(t_x)
            pre_lab = torch.argmax(output,dim=1)
            test_correct += torch.sum(pre_lab==t_y.data)
            test_num += t_x.size(0)
    test_acc = test_correct.double().item() / test_num
    print(f"准确率：{test_acc}")
if __name__ == '__main__':
    # LeNetP = LeNetP()
    # train_dataloader ,val_dataloader = train_val_data_process()
    # src = train_model(LeNetP,train_dataloader,val_dataloader,10)
    #
    tmodel = LeNetP()
    tmodel.load_state_dict(torch.load('./model/best_model5.pth'))
    test_dataloader = test_data_process()
    test_model(tmodel,test_dataloader)