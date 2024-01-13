import torch
import torch.nn as nn
import torch.functional as F
from torchsummary import summary

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(1,64,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        self.block2=nn.Sequential(
            nn.Conv2d(64,128,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.block3=nn.Sequential(
            nn.Conv2d(128,256,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.block4=nn.Sequential(
            nn.Conv2d(256,512,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )        
        self.block5=nn.Sequential(
            nn.Conv2d(512,512,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        ) 
        self.block6=nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512,4096),
            nn.Linear(4096,10)            
        ) 
    
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.block6(x)
        return x
    
if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=VGG16()
    model=model.to(device)
    print(summary(model, (1,224,224)))