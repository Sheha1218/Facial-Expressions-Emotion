import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 =nn.Conv2d(3,12,5,padding='same')
        self.pool =nn.MaxPool2d(2,2)
        self.conv2 =nn.Conv2d(12,24,5,padding='same')
        self.adopt_pool =nn.AdaptiveAvgPool2d((5,5))

        self.fc1 =nn.Linear(24*5*5,128)
        self.fc2 =nn.Linear(128,64)
        self.fc3=nn.Linear(64,16)
        self.fc4=nn.Linear(16,9)

    def forward(self,x):
      x=self.pool(F.relu(self.conv1(x)))
      x=self.pool(F.relu(self.conv2(x)))
      x=self.adopt_pool(x)
      x=torch.flatten(x,1)
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x

