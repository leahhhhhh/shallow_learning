import torch 
import torch.nn as nn
import torch.functional as F
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self,input_channel,output_channel,use_conv1=False):
        super(Residual,self).__init__()
        self.ReLU=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=output_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        self.b1=nn.BatchNorm2d(output_channel)
        self.b2=nn.BatchNorm2d(output_channel)
        if use_conv1:
            self.conv3=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        else:
            self.conv3=None
        
    def forward(self,x):
        y=self.ReLU(self.b1(self.conv1(x)))
        y=self.b1(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        y=self.ReLU(y+x)

        return y
    
# class Resnet18(nn.Module):



        

