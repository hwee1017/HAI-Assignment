import torch.nn as nn
import torch

class Model(nn.Module):
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
practice_model = Model()