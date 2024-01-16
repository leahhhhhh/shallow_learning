import torch 
from torch import nn 
from torchsummary import summary
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.c2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.f1=nn.Linear(400,120)
        self.f2=nn.Linear(120,84)
        self.f3=nn.Linear(84,10)

    def forward(self,x):
        x=F.relu(self.c1(x))
        x=F.max_pool2d(x,2)
        x=F.relu(self.c2(x))
        x=F.max_pool2d(x,2)
        x=x.view(x.size(0),-1)
        x=F.relu(self.f1(x))
        x=F.relu(self.f2(x))
        x=self.f3(x)
        return x 

if __name__ =="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model=Lenet().to(device)
    print(summary(model, input_size=(1,28,28)))






