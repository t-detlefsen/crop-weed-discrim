import torch.nn as nn

class ResNetCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Removing final FC layer and freezing weights - https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/5
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Define layers
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification logits in shape of (N, Nc)
        """
        
        out = self.resnet(x)
        out = out.squeeze()
        out = self.fc(out)

        return out