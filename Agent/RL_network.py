
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_FNN(nn.Module):
    """docstring for Net"""
    def __init__(self,J_num,O_max_len):
        super(CNN_FNN, self).__init__()
        # summary(self.conv1,(3,6,6))
        self.fc1 = nn.Linear(6*int(J_num/2)*int(O_max_len/2), 258)
        self.fc2 = nn.Linear(258,258)
        self.out = nn.Linear(258,17)

    def forward(self,x):
        x=self.conv1(x)
        x=x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class CNN_dueling(nn.Module):
    def __init__(self,J_num,O_max_len):
        super(CNN_dueling, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3,  #input shape (3,J_num,O_max_len)
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,  #使得出来的图片大小不变P=（3-1）/2,
            ),      # output shape (3,J_num,O_max_len)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,ceil_mode=False)  #output shape:  (6,int(J_num/2),int(O_max_len/2))
        )
        # summary(self.conv1,(3,6,6))
        self.val_hidden = nn.Linear(6*int(J_num/2)*int(O_max_len/2), 258)
        self.adv_hidden=nn.Linear(6*int(J_num/2)*int(O_max_len/2), 258)

        self.val=nn.Linear(258,1)
        self.adv = nn.Linear(258,17)

    def forward(self,x):
        x=self.conv1(x)
        x=x.view(x.size(0),-1)

        val_hidden = self.val_hidden(x)
        val_hidden = F.relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.relu(adv_hidden)

        val = self.val(val_hidden)
        adv = self.adv(adv_hidden)

        adv_ave = torch.mean(adv, dim=1, keepdim=True)
        x = adv + val - adv_ave
        return x