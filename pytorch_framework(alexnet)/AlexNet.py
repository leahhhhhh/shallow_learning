import torch 
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=0)
        self.c2=nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.c3=nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.c4=nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.c5=nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.f1=nn.Linear(9216,4096)
        # self.f2=nn.Linear(4096,4096)
        self.f3=nn.Linear(4096,10)



    def forward(self,x):
        x=F.relu(self.c1(x))
        x=F.max_pool2d(x,3,stride=2)
        x=F.relu(self.c2(x))
        x=F.max_pool2d(x,3,stride=2)
        x=F.relu(self.c3(x))
        x=F.relu(self.c4(x))
        x=F.relu(self.c5(x))
        x=F.max_pool2d(x,3,stride=2)
        x=x.view(x.size(0),-1)
        x=F.relu(self.f1(x))
        x=F.dropout(x, p=0.5)
        x=F.relu(self.f2(x))
        x=F.dropout(x,p=0.5)
        x=self.f3(x)
        return x

if __name__ =="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model=AlexNet().to(device)
    print(summary(model, input_size=(1,228,228)))     




