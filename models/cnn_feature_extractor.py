# models/cnn_feature_extractor.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class CNNFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, output_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        # Use ResNet18 backbone
        self.resnet = models.resnet18(pretrained=pretrained)
        # Remove final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Project to desired output dimension
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, x):
        # x: (batch_size, 3, H, W)
        features = self.resnet(x)  # (batch_size, 512, 1, 1)
        features = features.view(features.size(0), -1)  # flatten
        features = self.fc(features)  # (batch_size, output_dim)
        return features

# Example usage
if __name__ == "__main__":
    model = CNNFeatureExtractor()
    dummy_input = torch.randn(2,3,224,224)
    output = model(dummy_input)
    print(output.shape)  # should be (2, 128)
