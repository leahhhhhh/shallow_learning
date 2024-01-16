import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import precision_recall_fscore_support
import torchvision.transforms as transforms

sns.set()

def train(model, criterion, optimizer, training_dataloader):
    sum_accuracy = 0
    sum_loss = 0
    count = 0
    model.train()
    for features, labels in training_dataloader:
        #labels = torch.argmax(labels, dim=1)
        print('train_shape',labels.shape)
        probabilities=model(features)
        loss=criterion(probabilities,labels)
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mini_batch_size=labels.size(0)
        labels=torch.argmax(labels, dim=1)
        accuracy=(
            torch.sum(torch.max(probabilities,dim=1).indices==labels)
        )

        sum_accuracy+=accuracy.item()
        sum_loss+=loss.item()*mini_batch_size
        #print(sum_loss)
        count+=mini_batch_size
        #print(sum_loss/count)
        train_acc=sum_accuracy/count
        train_loss=sum_loss/count
    return train_acc, train_loss


def validate(model,criterion,dataloader):
    sum_accuracy=0
    sum_loss=0
    count=0
    model.eval()
    for features ,labels in dataloader:
        
        with torch.no_grad():
            probabilities=model(features)
            loss=criterion(probabilities,labels)
        mini_batch_size=labels.size(0)
        labels=torch.argmax(labels, dim=1)
        accuracy=(
            torch.sum(torch.max(probabilities,dim=1).indices==labels)
        )

        sum_accuracy+=accuracy.item()
        sum_loss+=loss.item()*mini_batch_size
        #print(sum_loss)
        count+=mini_batch_size
        #print(sum_loss/count)
        val_acc=sum_accuracy/count
        val_loss=sum_loss/count
    return val_acc, val_loss
    
def evaluate(dataset,model):
    # x_test=dataset["Xts"]
    # y_test=dataset["Yts"]
    # x_test=torch.from_numpy(x_test.astype(np.float32) / 255).permute(0, 3, 1, 2)
    # cifar_xtest=torch.Tensor(x_test)
    # print(cifar_xtest.shape)
    # y_pred=torch.max(model(cifar_xtest),dim=1).indices
    # correct=0
    # for index , value in enumerate(y_pred):
    #     if value == y_test[index]:
    #         correct += 1
    # #report=classification_report(y_test,y_pred.numpy())
    # precision,recall,_,_=precision_recall_fscore_support(y_test,y_pred,average='weighted')

    correct=0
    sum_count=0
    precision_all=0
    recall_all=0
    times=0
    model.eval()
    for features , labels in dataset:
        print(features,labels)
        with torch.no_grad():
            probabilities=model(features)
            print('probailities',probabilities)
            y_pred=torch.max(probabilities,dim=1).indices
            for index , value in enumerate(y_pred):
                if value == labels[index]:
                    correct += 1
            sum_count+=len(y_pred)
            precision,recall,_,_=precision_recall_fscore_support(labels,y_pred,average='weighted')
            precision_all+=precision
            recall_all+=recall
            times+=1
    
    accuracy=correct/sum_count
    precision=precision_all/times
    recall=recall_all/times

    return precision, recall,accuracy


        
