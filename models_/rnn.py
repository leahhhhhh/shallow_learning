import torch 
import torch.nn as nn
import torchvision

num_time_steps=16
input_size=3
hidden_size=16
output_size=3
num_layers=1
lr=0.01

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN,self).__init__()
        self.rnn=nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0.0, std=0.001)
        self.linear=nn.Linear(hidden_size,output_size)
    # def forward(self,x,hidden_prev):


