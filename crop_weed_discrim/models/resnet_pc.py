import torch.nn as nn
import torchvision

class ResNetCNN(nn.Module):
    """
    Model definition with feature extraction
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Removing final FC layer and freezing weights
        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])  # Up to penultimate layer
        self.fc = nn.Linear(512, num_classes)  # Classifier

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: features: extracted features from the penultimate layer
                 logits: classification logits
        """
        features = self.feature_extractor(x)
        features = features.squeeze()  # Flatten features
        logits = self.fc(features)  # Compute class logits
        return features, logits