import copy
import time

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import pandas as pd


from model import AlexNet

def train_val_data_process():
    train_dataset = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
                              download=True
                              )
    train_data,val_data = Data.random_split(train_dataset,[round(0.8*len(train_dataset)),round(0.2*len(train_dataset))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                   batch_size=8,
                                   shuffle=True,
                                   num_workers=8)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=8,
                                     shuffle=True,
                                     num_workers=8)
    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    # 损失函数 (交叉熵损失)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []


    for epoch in range(num_epochs):
        # batch time
        since = time.time()
        print("-"*10)
        print(f"Epoch {epoch+1}/{num_epochs}")

        # 初始化参数
        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        train_num = 0
        val_num = 0

        for step,(b_x,b_y) in enumerate(train_dataloader):
            print(f'Epoch {epoch},Train Step {step}')
            # 将特征放入训练设备
            b_x = b_x.to(device)
            # 将标签放入训练设备
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()
            # 前向传播过程，输入一个batch，输出一个batch对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的下标
            pre_lab = torch.argmax(output,dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output,b_y)

            # 梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 累加损失函数
            train_loss += loss.item() * b_x.size(0)
            # 计算准确度
            train_corrects += torch.sum(pre_lab==b_y.data)

            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            print(f'Epoch {epoch},Val Step {step}')
            # 将特征放入训验证设备
            b_x = b_x.to(device)
            # Put label into evaluate device
            b_y = b_y.to(device)
            # Set model into evaluation models
            model.eval()
            # forward, input a batch
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            loss = criterion(output,b_y)

            val_loss += loss.item()*b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)
        # calculate and save each epochs' loss and accuracy
        train_loss_all.append(train_loss / train_num)
        # item: change tensor into data which python can calculate
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))

        if val_acc_all[-1]>best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("Using time:{:.4f} m {:.4}s".format(time_use//60,time_use%60))

    import os
    folder_path = "./model"
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])+1

    torch.save(best_model_wts, f"./model/best_model{file_count}.pth")

    train_process = pd.DataFrame(data={
        "epoch":range(num_epochs),
        "train_loss_all":train_loss_all,
        "val_loss_all":val_loss_all,
        "train_acc_all":train_acc_all,
        "val_acc_all":val_acc_all
    })
    return train_process, f"./model/best_alex_model{file_count}.pth"

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    # one row,two list,first img
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,'bs-',label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    AlexNet = AlexNet()
    train_dataloader, val_dataloader =train_val_data_process()
    train_process,src = train_model_process(AlexNet,train_dataloader,val_dataloader,6)
    matplot_acc_loss(train_process)
    print(f"Finish:{src}")