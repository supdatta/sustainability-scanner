import torch.nn as nn
import torch.nn.functional as F

class SustainabilityCNN(nn.Module):
    def __init__(self, num_classes):
        super(SustainabilityCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 3 x 128 x 128
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 x 64 x 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 x 32 x 32
        x = self.pool(F.relu(self.conv3(x)))  # 128 x 16 x 16
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
