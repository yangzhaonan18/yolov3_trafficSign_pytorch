import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),      #(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  #output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
 
    # ¶¨ÒåÇ°Ïò´«²¥¹ý³Ì£¬ÊäÈëÎªx
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
 
dummy_input = torch.rand(13, 1, 28, 28)  
model = LeNet()

with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input, ))
