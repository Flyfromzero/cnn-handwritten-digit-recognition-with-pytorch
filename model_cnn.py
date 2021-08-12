import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)#kernal size is 5 * 5
        self.pool1 = nn.MaxPool2d(2, 2)#pooling window is 2 * 2
        self.conv2 = nn.Conv2d(10, 20, 5)#kernal size is 5 * 5
        self.pool2 = nn.MaxPool2d(2, 2)#pooling window is 2 * 2
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 320)#reshape x into u * 320, u is unknown
        x = self.fc2(self.fc1(x))
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)