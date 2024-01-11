import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetFeatureExtractor(nn.Module):
    """
    ResNetFeatureExtractor extracts Image features using a pre-trained ResNet-50.
    """

    def __init__(self):
        """
        Creates an instance of ResNetFeatureExtractor.

        It loads a pre-trained ResNet-50, removes the last fully connected layer,
        and sets the model in evaluation mode.
        """
        super(ResNetFeatureExtractor, self).__init__()

        # Load the pretrained ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the last FC layer [2048 x 1000]
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Set the model on evaluation mode
        self.feature_extractor.eval()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ResNetFeatureExtractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Extracted features tensor with shape (batch_size, 1, feature_dim).
        """
        with torch.no_grad():
            batch = x.size(0)
            features = self.feature_extractor(x)
            features = features.view(batch, 1, -1)  # [batch, 1, 2048]

        return features
