import torch.nn as nn
import torchvision

class ResNetCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Removing final FC layer and freezing weights - https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/5
        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # Define layers
        self.fc = nn.Linear(512, num_classes)

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