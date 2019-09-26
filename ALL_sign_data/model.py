import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
 


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, classes)

        # Dropout module with 0.2 drop probability
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x




class Lenet5(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv = torch.nn.Sequential(
           
         
            torch.nn.Conv2d(3, 6, 5), # out = (in - kernal + 2*padding) / stride + 1  (1, 28, 28) -> (6, 24, 24)
            torch.nn.ReLU(), 
            torch.nn.AvgPool2d(2, stride=2), # (N, 6, 24, 24) - > (N, 6, 12, 12)
            
            torch.nn.Conv2d(6, 16, 5),  #  (N, 6, 12, 12) -> (N. 6, 8, 8)
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, stride=2)  # (N, 16, 8, 8) -> (N, 16, 4 4)
        )
        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(256, 120), #(N, 256) - > (N, 120)
            # torch.nn.ReLU(),
            # torch.nn.Linear(120, 84), #(N, 120) - > (N, 84)
            # torch.nn.ReLU(),
            # torch.nn.Linear(84, classes)

            # torch.nn.Linear(576, 256),
            # torch.nn.ReLU(),
           
            torch.nn.Linear(256, classes),
            torch.nn.ReLU(),
            torch.nn.Linear(classes, classes),
            
            )
    def forward(self, x):
        x = self.conv(x)
        # print(x.size()) # torch.Size([64, 16, 4, 4])
        x = x.view(x.size(0), -1)  # batch_size is x.size(0)
      
        x = self.fc(x)
        return x




def my_resnt18(classes):
    resnet = resnet18()
    resnet.fc = torch.nn.Linear(in_features=512, out_features=classes, bias=True)
    resnet.add_module("softmax", torch.nn.Softmax(dim=-1))

    return resnet




# classes =145
# myresnt18 = resnet18()
# myresnt18.fc = torch.nn.Linear(in_features=512, out_features=classes, bias=True)
# myresnt18.add_module("softmax", torch.nn.Softmax(dim=-1))


class FashionCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=300)  #   classes: 300 to 120 to 10
        self.fc3 = nn.Linear(in_features=300, out_features=classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

