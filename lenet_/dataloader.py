from torchvision import transforms
from PIL import Image
from lenet import Lenet
import torch

def dataloader():

    train_transform=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize([0.162,0.151,0.138],[0.058,0.052,0.048]),
        transforms.Grayscale(1)
    ])
    image = Image.open('/Users/jiawen/Desktop/test.jpg')
    image = image.convert('RGB')
    image=train_transform(image)
    print(image.shape)
    image=image.unsqueeze(0)
    print(image.shape)
    return image


if __name__=="__main__":
    model=Lenet()
    model.load_state_dict(torch.load('pytorch_framework(lenet)/best_model.pth'))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    with torch.no_grad():
        model.eval()
        image=dataloader()
        image=image.to(device)
        output=model(image)
        prelabel=torch.argmax(output,dim=1)
        print(prelabel.item())
        
