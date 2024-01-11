import torch
import torch.nn as nn
import torch.nn.functional as F


class Tnet(nn.Module):
    """
    TNet learns the Transformation matrix with a specified dimensions
    to rotate the input point cloud to a consistent orientation.
    """

    def __init__(self, dim: int, num_points: int):
        """
        Creates an instance of Tnet.

        Args:
            dim (int): The dimension of the input point cloud.
            num_points (int): The number of points in the input point cloud.
        """
        super(Tnet, self).__init__()
        self.dim = dim

        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=dim * dim)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Tnet.

        Args:
            x (torch.Tensor): Input point cloud tensor of shape (batch_size, dim, num_points).

        Returns:
            torch.Tensor: Transformation matrix applied of shape (batch_size, dim, dim).
        """
        batch = x.size(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # x = self.max_pool(x).view(batch, -1)  # [batch, 1024, 1] -> [batch, 1024]
        x = torch.max(x, 2, keepdim=True)[0].view(
            batch, -1
        )  # [batch, 1024, 1] -> [batch, 1024, 1] -> [batch, 1, 1]
        # print(f"x.shape: {x.shape}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [batch, dim*dim]
        x = x.view(batch, self.dim, self.dim)  # [batch, dim, dim]

        # Add the identity matrix to the output
        iden = torch.eye(self.dim, requires_grad=True).repeat(batch, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden

        return x
