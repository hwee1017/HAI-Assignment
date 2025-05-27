import torch
import torch.nn as nn
from practice import practice_model

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(practice_model.parameters(),lr=0.01)
epchs = 100
for epch in range(1,epchs+1):
    inputs = torch.randn(1,17)
    labels = torch.randn(1,1)
    optimizer.zero_grad()
    outputs = practice_model(inputs)
    loss = loss_function(outputs,labels)
    loss.backward()
    optimizer.step()

print(f"epch : {epch},loss : {loss.item():.4f}")