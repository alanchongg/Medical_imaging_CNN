import torch.nn as nn
import torch.nn.functional as F

class Cnn4A1LR1(nn.Module):
    def __init__(self):
        super(Cnn4A1LR1, self).__init__()
        self.con1 = nn.Conv2d(3, 16, 3)
        self.con2 = nn.Conv2d(16, 32, 3)
        self.con3 = nn.Conv2d(32, 64, 3)
        self.con4 = nn.Conv2d(64, 128, 3)
        self.fapool = nn.AvgPool2d(3, 3)
        self.apool = nn.AvgPool2d(2, 2)
        self.mpool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(128*10*10, 9600)
        self.linear2 = nn.Linear(9600, 7200)
        self.linear3 = nn.Linear(7200, 3)
        self.relu=nn.ReLU(inplace=True)
        self.lrelu=nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.fapool(self.lrelu(self.con1(x)))
        x = self.mpool(self.relu(self.con2(x)))
        x = self.mpool(self.relu(self.con3(x)))
        x = self.mpool(self.relu(self.con4(x)))
        x = x.view(-1, 128*10*10)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        # x = F.softmax(x, dim=1)
        return x