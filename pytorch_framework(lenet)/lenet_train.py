from lenet import Lenet
from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import copy
import pandas as pd

def train_val_process():
    train_set=FashionMNIST(root='./data',
                         train=True,
                         transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                         download=True)

    train_data,val_data=train_test_split(train_set,test_size=0.2,shuffle=True)
    train_loader=DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)
    val_loader=DataLoader(dataset=val_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)
    ###
    test_set=FashionMNIST(root='./data',
                            train=False,
                            transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                            download=True)
    test_loader=DataLoader(dataset=test_set,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0)
    
    return train_loader,val_loader, test_loader



# train_val_process()
def train_model_process(model,train_loader,val_loader,test_loader,epochs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimize=torch.optim.SGD(params=model.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss()
    model=model.to(device)
    best_model_wts=copy.deepcopy(model.state_dict())

    best_acc=0.0
    train_loss_all=[]
    val_loss_all=[]
    train_acc_all=[]
    val_acc_all=[]
    test_loss_all=[]
    test_acc_all=[]

    since=time.time()

    for epoch in range(epochs):
        print("epoch",epoch)
        print("-"*20)
        train_acc=0.0
        train_loss=0.0
        val_acc=0.0
        val_loss=0.0
        train_num=0
        val_num=0
        for step,(b_x,b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            ##
            model.train()
            output=model(b_x)
            #pre_label=torch.argmax(output,dim=1).float().requires_grad_()
            # b_y=b_y.float()
            loss=criterion(output,b_y)
            ##initial gradient
            optimize.zero_grad()
            loss.backward()
            optimize.step()
            train_loss+=loss.item()*b_x.size(0)
            pre_label = torch.argmax(output, dim=1)
            train_acc+=torch.sum(pre_label==b_y.data)
            train_num+=b_x.size(0)
            if step%100==0 :
                loss,current=loss.item(),(step+1)*len(b_x)
                print(f"loss:{loss:.4f}")
    
        for step,(b_x,b_y) in enumerate(val_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            ##
            model.eval()
            output=model(b_x)
            # pre_label=torch.argmax(output,dim=1).float().requires_grad_()
            output = output
            loss=criterion(output,b_y)
            ##initial gradient
            optimize.zero_grad()
            loss.backward()
            optimize.step()
            val_loss+=loss.item()*b_x.size(0)
            pre_label = torch.argmax(output,dim=1)
            val_acc+=torch.sum(pre_label==b_y.data)
            val_num+=b_x.size(0)

        test_acc, test_loss, test_num= test(test_loader,model)

        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_acc.double().item()/train_num)
        val_acc_all.append(val_acc.double().item()/val_num)


        print("{} train loss : {:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1],train_acc_all[-1]))
        print("{} val loss : {:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1],val_acc_all[-1]))

        test_acc_all.append(test_acc.double().item()/test_num)
        test_loss_all.append(test_loss/test_num)
        print("test loss: {:.4f}, test acc {:.4f}".format(test_loss_all[-1],test_acc_all[-1]))



        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
        time_use=time.time() - since
        since=time.time()
        print("time use:",time_use)

    torch.save(best_model_wts,f"/Users/jiawen/Desktop/USYD_S2/5328/assignment2/best_model.pth")

    train_process=pd.DataFrame(data={"epoch":range(epochs),
                                     "train_loss_all":train_loss_all,
                                     "val_loss_all":val_loss_all,
                                     "train_acc_all":train_acc_all,
                                     "val_acc_all":val_acc_all,
                                     })
    
    test_process=pd.DataFrame(data={"epoch":range(epochs),
                                    "test_loss_all":test_loss_all,
                                    "test_acc_all":test_acc_all})
    return train_process,test_process


def test(dataloader,model):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    test_loss=0.0
    test_acc=0.0
    test_num=0
    test_loss_all=[]
    test_acc_all=[]
    # for epoch in epochs:
    #     print("epoch",epoch)
    #     print("-"*20)
    with torch.no_grad():
        for _, (b_x,b_y) in enumerate(dataloader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            model.eval()
            output=model(b_x)
            loss=criterion(output,b_y)
            pre_label=torch.argmax(output,dim=1)
            test_loss+=loss.item()*b_x.size(0)
            test_acc+=torch.sum(pre_label==b_y)
            test_num+=b_x.size(0)
    # test_acc_all.append(test_acc.double().item()/test_num)
    # test_loss_all.append(test_loss/test_num)
    # print("test loss: {:.4f}, test acc {:.4f}".format(test_loss_all[-1],test_acc_all[-1]))

    return test_acc, test_loss, test_num




def plot(train_process,test_process):
    plt.figure(figsize=(12,4))
    plt.subplot(2,2,1)
    plt.plot(train_process['epoch'],train_process.train_loss_all,"ro-",label='train loss')
    plt.plot(train_process['epoch'],train_process.val_loss_all,"bs-",label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(2,2,2)
    plt.plot(train_process['epoch'],train_process.train_acc_all,"ro-",label='train acc')
    plt.plot(train_process['epoch'],train_process.val_acc_all,"bs-",label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.subplot(2,2,3)
    plt.plot(test_process['epoch'],test_process.test_loss_all,"ro-",label='test loss')
    #plt.plot(train_process['epoch'],train_process.val_acc_all,"bs-",label='train acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
   
    plt.subplot(2,2,4)
    plt.plot(test_process['epoch'],test_process.test_acc_all,"ro-",label='test acc')
    #plt.plot(train_process['epoch'],train_process.val_acc_all,"bs-",label='train acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')


    plt.savefig('./test.png')


if __name__ == "__main__":
    Lenet=Lenet()
    train_loader,val_loader,test_loader=train_val_process()
    train_process,test_process=train_model_process(Lenet,train_loader,val_loader,test_loader,4)
    plot(train_process,test_process)


