import hydra
from hydra.utils import instantiate

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pointfusion.modules.resnet import ResNetFeatureExtractor
from pointfusion.modules.pointnet import PointNetEncoder
from pointfusion.losses.loss import (
    global_fusion_net_loss,
    dense_fusion_net_loss,
    get_corner_offsets_from_pc,
    get_corners_from_pred_offsets,
)
from pointfusion.utils.objectron import iou
from pointfusion.utils.iou_utils import np_get_3d_bounding_box_including_center
from pointfusion.utils.draw_utils import draw_bounding_boxes


class PointFusionLit(pl.LightningModule):
    """PointFusionLit is a PyTorch Lightning module for point cloud and image features fusion."""

    def __init__(
        self, num_points: int, fusion_type: str, lr: float, draw_bbox: bool = False
    ):
        """
        Creates an instance of the PointFusionLit.

        Args:
            num_points (int): Number of points in the point cloud.
            fusion_type (str): Type of fusion, either "global" or "dense".
            lr (float): Learning rate for optimization.
            draw_bbox (bool): Whether to draw the bounding boxes or not during testing.
                              if True, the bounding boxes will be drawn in the for first
                              batch in the test set.
        """
        super(PointFusionLit, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.fusion_type = fusion_type
        self.draw_bbox = draw_bbox

        self.resnet = ResNetFeatureExtractor()
        self.pointnet = PointNetEncoder(num_points=num_points, num_global_features=1024)

        # Define MLP layers
        if self.fusion_type == "global":
            self.fc1 = nn.Linear(in_features=3072, out_features=512)  # Global Fusion
        if self.fusion_type == "dense":
            self.fc1 = nn.Linear(in_features=3136, out_features=512)  # Dense Fusion

        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)

        self.fc4 = nn.Linear(in_features=128, out_features=8 * 3)  # BBox Regressor
        self.fc5 = nn.Linear(in_features=128, out_features=1)  # Confidence Score

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img: torch.Tensor, pc: torch.Tensor):
        """
        Forward pass of the PointFusionLit.

        Args:
            img (torch.Tensor): Input image tensor.
            pc (torch.Tensor): Input point cloud tensor.

        Returns:
            torch.Tensor: Predicted corners or corner offsets based on the fusion type.
        """
        batch, dim, num_points = pc.size()

        if self.fusion_type == "global":
            # Extract features from image
            image_features = self.resnet(img)  # (batch, 1, 2048)

            # Extract features from point cloud
            local_features, global_features, mat_64x64 = self.pointnet(pc)
            global_fusion = torch.cat(
                (image_features, global_features), dim=2
            )  # (batch, num_points, 3072)

            # Pass through MLP
            x = self.relu(self.fc1(global_fusion))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            pred_corners = self.fc4(x)
            pred_corners = pred_corners.view(batch, 8, 3)  # (batch, num_points, 8, 3)

            return pred_corners, mat_64x64

        elif self.fusion_type == "dense":
            # Extract features from image
            image_features = self.resnet(img)
            image_features = image_features.repeat(
                1, num_points, 1
            )  # (batch, num_points, 2048)

            # Extract features from point cloud
            local_features, global_features, mat_64x64 = self.pointnet(pc)
            global_features = global_features.repeat(
                1, num_points, 1
            )  # (batch, num_points, 1024)
            dense_features = torch.cat(
                (image_features, global_features, local_features), dim=2
            )  # (batch, num_points, 3136)

            # Pass through MLP
            x = self.relu(self.fc1(dense_features))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            corner_offsets = self.fc4(x)
            corner_offsets = corner_offsets.view(
                batch, num_points, 8, 3
            )  # (batch, num_points, 8, 3)

            # Confidence Scores
            scores = self.fc5(x)
            scores = self.softmax(scores).view(
                batch, -1
            )  # (batch, num_points) probability of each point being a corner

            return corner_offsets, scores, mat_64x64

    def training_step(self, batch, batch_idx):
        """
        Training step for the PointFusionLit model.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        images = batch["img_t"]
        pcs = batch["pc_t"]
        target = batch["bbox_t"]

        if self.fusion_type == "global":
            # Forward pass
            pred_corners, mat_64x64 = self(images, pcs)
            loss = global_fusion_net_loss(pred_corners, target, mat_64x64)

            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

        elif self.fusion_type == "dense":
            # Forward pass
            pred_corners_offsets, scores, mat_64x64 = self(images, pcs)

            # Convert target into corner offsets
            target = get_corner_offsets_from_pc(target, pcs.permute(0, 2, 1))
            loss = dense_fusion_net_loss(
                pred_corners_offsets, scores, target, mat_64x64
            )
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the PointFusionLit model.

        Args:
            batch: Validation batch.
            batch_idx: Batch index.
        """
        # images = batch['images']
        # pcs = batch['pcs']
        # target = batch['bboxes']
        images = batch["img_t"]
        pcs = batch["pc_t"]
        target = batch["bbox_t"]

        if self.fusion_type == "global":
            # Forward pass
            pred_corners, _ = self(images, pcs)
            print(pred_corners.shape)
            print(target.shape)
            pred_corners = np_get_3d_bounding_box_including_center(
                pred_corners[0].cpu().numpy()
            )
            target = np_get_3d_bounding_box_including_center(target[0].cpu().numpy())
            iou_3d = iou.IoU(pred_corners, target)
            self.log(
                "global_val_iou",
                iou_3d.iou(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        elif self.fusion_type == "dense":
            # Forward pass
            pred_corners_offsets, scores, _ = self(images, pcs)
            pred_corners = get_corners_from_pred_offsets(
                pred_corners_offsets, pred_scores=scores
            )
            iou_3d = iou.IoU(pred_corners, target)
            self.log(
                "dense_val_iou",
                iou_3d.iou(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        """
        Test step for the PointFusionLit model.

        Args:
            batch: Test batch.
            batch_idx: Batch index.
        """
        images = batch["img_t"]
        pcs = batch["pc_t"]
        target = batch["bbox_t"]

        if self.fusion_type == "global":
            # Forward pass
            pred_corners, _ = self(images, pcs)
            pred_corners_for_iou = np_get_3d_bounding_box_including_center(
                pred_corners[0].cpu().numpy()
            )
            target_for_iou = np_get_3d_bounding_box_including_center(
                target[0].cpu().numpy()
            )
            iou_3d = iou.IoU(pred_corners_for_iou, target_for_iou)
            self.log(
                "global_test_iou",
                iou_3d.iou(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            if self.draw_bbox and batch_idx == 0:
                pred_corners = pred_corners.cpu().numpy()
                pcs = pcs.cpu().numpy()
                target = target.cpu().numpy()
                pred_corners = pred_corners[0]
                pcs = pcs[0].transpose(1, 0)
                target = target[0]
                draw_bounding_boxes(pcs, target)
                draw_bounding_boxes(pcs, pred_corners)

        elif self.fusion_type == "dense":
            # Forward pass
            pred_corners_offsets, scores, _ = self(images, pcs)
            pred_corners = get_corners_from_pred_offsets(
                pred_corners_offsets, pred_scores=scores
            )
            pred_corners_for_iou = np_get_3d_bounding_box_including_center(
                pred_corners[0].cpu().numpy()
            )
            target_for_iou = np_get_3d_bounding_box_including_center(
                target[0].cpu().numpy()
            )
            iou_3d = iou.IoU(pred_corners_for_iou, target_for_iou)
            self.log(
                "dense_test_iou",
                iou_3d.iou(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            if self.draw_bbox and batch_idx == 0:
                pred_corners = pred_corners.cpu().numpy()
                pcs = pcs.cpu().numpy()
                target = target.cpu().numpy()
                pred_corners = pred_corners[0]
                pcs = pcs[0].transpose(1, 0)
                target = target[0]
                draw_bounding_boxes(pcs, target)
                draw_bounding_boxes(pcs, pred_corners)

    def configure_optimizers(self):
        """
        Configure the optimizer for the PointFusionLit model.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
