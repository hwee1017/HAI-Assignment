import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset = datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
test_dataset = datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN,self).__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.conv_seq(x)
        return self.fc_seq(x)
model = myCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epochs = 5
for epoch in range(1,epochs+1):
    model.train()
    total_loss = 0.0
    for images, labels in train_dataloader:
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_dataloader):.4f}")

img = Image.open("C:/Users/user/OneDrive/바탕 화면/HAI-Assignment/assignment7/shoesImage.png").convert('L')
to_tensor = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
model.eval()
x = to_tensor(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    probs = nn.functional.softmax(logits,dim=1)
print(probs)