import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from lenet import Lenet
from torch.utils.data import DataLoader

def test_data_process():
    test_data=FashionMNIST(root='./data',
                            train=False,
                            transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                            download=True)
    test_loader=DataLoader(dataset=test_data,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0)

    return test_loader

# test_loader=test_data_process()

def test_model_process(model,test_loader):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    test_acc=0.0
    test_nums=0
    with torch.no_grad():
        for b_x,b_y in test_loader:
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            model.eval()
            output=model(b_x)
            # loss=(output,b_y)
            pre_label=torch.argmax(output,dim=1)
            test_acc+=torch.sum(pre_label==b_y.data)
            test_nums+=b_x.size(0)
    test_acc_f=test_acc.double().item()/test_nums
    print("test acc {:.4f}".format(test_acc_f))


if __name__=="__main__":
    model=Lenet()
    model.load_state_dict(torch.load('/Users/jiawen/Desktop/USYD_S2/5328/assignment2/best_model.pth'))
    test_loader=test_data_process()
    test_model_process(model,test_loader)




