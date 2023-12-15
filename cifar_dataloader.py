from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

class DatasetArray(Dataset):
    def __init__(self,features,labels,transform):
        self.features=features
        self.labels=labels
        self.transform=transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,index):
        features=self.features[index]
        labels=self.labels[index]
        #features = Image.fromarray(features.astype(np.uint8))
        features_np = features.numpy()

        # Convert NumPy array to PIL Image
        features= Image.fromarray((features_np * 255).astype('uint8').transpose(1, 2, 0))
        
        if self.transform is not None:
            features=self.transform(features)
            #print(features.shape)
        return (features,labels)
    

def dataloader2(dataset):
    train_features=dataset['Xtr']
    #train_features=train_features
    train_labels=dataset['Str']
    train_labels = torch.from_numpy(train_labels)
    train_labels=torch.nn.functional.one_hot(train_labels, num_classes=3).float()
    test_features=dataset['Xts']
    test_labels=dataset['Yts']
    test_labels = torch.from_numpy(test_labels)
    test_labels=torch.nn.functional.one_hot(test_labels, num_classes=3).float()
    x_train,x_val,y_train,y_val=train_test_split(train_features[:500],train_labels[:500],test_size=0.2)
    print(x_train.shape)
    x_train=torch.from_numpy(x_train.astype(np.float32) / 255).permute(0, 3, 1, 2)
    x_val=torch.from_numpy(x_val.astype(np.float32) / 255).permute(0, 3, 1, 2)
    x_test=torch.from_numpy(test_features.astype(np.float32) / 255).permute(0, 3, 1, 2)
    #batch load
    data_transform = {
    "train": transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    #transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

    "test":transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor()])}
    train_dataset=DatasetArray(x_train,y_train,transform=data_transform["train"])
    val_dataset=DatasetArray(x_val,y_val,transform=data_transform["val"])
    # train_dataset=DatasetArray(x_train,y_train)
    # val_dataset=DatasetArray(x_val,y_val)
    test_data=DatasetArray(x_test,test_labels,transform=data_transform["test"])
    
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=64,shuffle=True)
    test_loader=DataLoader(test_data,batch_size=64,shuffle=True)

    return train_loader,val_loader,test_loader

