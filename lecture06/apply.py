import torch
import torch.nn as nn

class Mine(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequence = nn.Sequential(
            nn.Linear(17,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1)
        )
    def forward(self,x):
        x = self.flatten(x)
        y = self.sequence(x)
        return y

firstmodel = Mine()
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(firstmodel.parameters(),lr=0.01)
Epchs = 100
for epch in range(1,Epchs+1):
    inputs = torch.randn(1,17)
    labels = torch.randn(1,1)
    outputs = firstmodel(inputs)
    loss = loss_function(outputs,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"epch : {epch} loss : {loss.item():.4f}")