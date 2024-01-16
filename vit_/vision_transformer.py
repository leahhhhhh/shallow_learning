import torchvision.models as models
import numpy  as np
from vit_.cifar_dataloader import dataloader1
from train_evaluate import train,validate,evaluate
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
sns.set()

CIFAR = np.load('./datasets/CIFAR.npz')
train_loader,val_loader,test_loader=dataloader1(CIFAR)

weights = models.ViT_B_16_Weights.DEFAULT
# vit = models.vit_b_16(weights=weights)
# num_ftrs = vit.config.num_labels
# vit.head = nn.Linear(num_ftrs, 3)

class VIT(nn.Module):
    def __init__(self, output_shape):
        super(VIT, self).__init__()

        self.pretrained = models.vit_b_16(weights=weights)
        for param in self.pretrained.parameters():
            param.requires_grad = False

        pretrained_output = 1000
        self.fc = torch.nn.Sequential(
            nn.Linear(in_features=pretrained_output,
                      out_features=output_shape,
                      bias=False),
        )

    def forward(self, image):
        output = self.pretrained(image)
        output = self.fc(output)
        return output

vit = VIT(output_shape=3)



def main(dataset,epochs=10,lr=0.01):

    train_loader,val_loader,test_loader=dataloader1(dataset)
    #model=resnet18
    #model=Lenet()
    model=vit
    # resnet18 = resnet18.fc.in_features
    # resnet18.fc = nn.Linear(num_ftrs, 3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        train_acc, train_loss=train(model,criterion, optimizer, train_loader)
        val_acc, val_loss=validate(model,criterion, val_loader)
        test_acc, test_loss=validate(model,criterion, test_loader)

        training_accuracies.append(train_acc)
        training_losses.append(train_loss)
        validation_accuracies.append(val_acc)
        validation_losses.append(val_loss)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)


        print(f"epoch: {epoch}\n \
            train_acc: {train_acc}, train_loss: {train_loss}\n \
            val_acc: {val_acc}, val_loss: {val_loss}\n \
            test_acc: {test_acc}, test_loss: {test_loss}\n")
    precision,recall=evaluate(dataset,model)


    return (training_accuracies,training_losses,
            validation_accuracies,
            validation_losses,
            test_accuracies,
            test_losses,
            precision,
            recall,
            model)



if __name__ == '__main__':
    (training_accuracies,training_losses,
    validation_accuracies,
    validation_losses,
    test_accuracies,
    test_losses,
    precision,
    recall,
    model)=main(CIFAR,epochs=2,lr=0.01)
