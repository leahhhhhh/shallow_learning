from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_data=FashionMNIST(root='./data',
                         train=True,
                         transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                         download=True)
train_loader=DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)

for step, (b_x,b_y) in enumerate(train_loader):
    if step > 0:
       break

batch_x=b_x.squeeze().numpy()## 64*224*224*1->64*224*224
batch_y=b_y.numpy()
print(batch_y)
calss_label=train_data.classes
print(calss_label)

plt.figure(figsize=(12,5))
for i in np.arange(len(batch_y)):
    plt.subplot(4,16,i+1)
    plt.imshow(batch_x[i,:,:],cmap=plt.cm.gray)

plt.show()
