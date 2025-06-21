import torch.nn as nn
import torchvision.models as models

class SustainabilityCNN(nn.Module):
    def __init__(self, num_classes):
        super(SustainabilityCNN, self).__init__()
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

