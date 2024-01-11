import torch
import torch.nn as nn
import torch.nn.functional as F

from pointfusion.modules.tnet import Tnet


class PointNetEncoder(nn.Module):
    """PointNetEncoder module for encoding point clouds in latent space."""

    def __init__(self, num_points: int, num_global_features: int):
        """
        Create an instance of the PointNetEncoder.

        Args:
            num_points (int): The number of points in the input point cloud.
            num_global_features (int): The number of global features to output.
        """
        super(PointNetEncoder, self).__init__()

        self.num_points = num_points
        self.num_global_features = num_global_features

        # Pass through T-Net (3x3)
        self.tnet3 = Tnet(dim=3, num_points=num_points)
        self.tnet64 = Tnet(dim=64, num_points=num_points)

        # MLP(64, 64)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)

        # MLP(64, 128, 1024)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(
            in_channels=128, out_channels=self.num_global_features, kernel_size=1
        )

        # Max Pooling
        self.max_pool = nn.MaxPool1d(kernel_size=self.num_points)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the PointNetEncoder.

        Args:
            x (torch.Tensor): Input point cloud tensor of shape (batch_size, num_points, 3).

        Returns:
            tuple: A tuple containing local features, global features, and the 64x64 transformation matrix.
        """
        # Pass through T-Net (3x3)
        mat_3x3 = self.tnet3(x)

        # Perform transformation using T-Net output
        x = x.transpose(2, 1)
        x = torch.bmm(x, mat_3x3).transpose(2, 1)

        # Pass through MLP(64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # (batch, 64, num_points)

        # Pass through T-Net (64x64)
        mat_64x64 = self.tnet64(x)

        # Perform transformation using T-Net output
        x = x.transpose(2, 1)
        x = torch.bmm(x, mat_64x64).transpose(2, 1)
        local_features = x.clone().permute(0, 2, 1)  # (batch, num_points, 64)

        # Pass through MLP(64, 128, 1024)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)  # (batch, 1024, num_points)

        # Max Pooling
        x = self.max_pool(x)  # (batch, 1024, 1)
        global_features = x.permute(0, 2, 1)  # (batch, 1, 1024)
        return local_features, global_features, mat_64x64
