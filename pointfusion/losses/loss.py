import numpy as np
import torch
import torch.nn as nn


def stn_loss(transformation_mat: torch.Tensor) -> torch.Tensor:
    """
    Spatial transformation regularization loss introduced in PointNet to enforce the
    orthogonality to the learned spatial transform matrix.

    Args:
        transformation_mat (torch.Tensor): Transformation matrix tensor.

    Returns:
        torch.Tensor: Regularization loss.
    """
    batch, dim, dim = transformation_mat.size()
    iden = torch.eye(dim, requires_grad=True)[None, :, :]
    if transformation_mat.is_cuda:
        iden = iden.cuda()

    loss = torch.mean(
        torch.norm(
            torch.bmm(transformation_mat, transformation_mat.transpose(2, 1) - iden),
            dim=(1, 2),
        )
    )
    return loss


def global_fusion_net_loss(pred_corners, target_corners, mat_64x64) -> torch.Tensor:
    """
    Calculate global fusion network loss, which is sum of corners loss and spatial
    transformation regularization loss.

    Args:
        pred_corners (torch.Tensor): Predicted corners tensor.
        target_corners (torch.Tensor): Target corners tensor.

    Returns:
        torch.Tensor: Total global fusion loss.
    """
    device = pred_corners.device  # Get the device of the input tensors

    # Corners loss
    L1 = nn.SmoothL1Loss(reduction="mean")
    loss_corner = L1(pred_corners, target_corners)  # [B]
    loss_corner = torch.mean(loss_corner)

    # STN Loss
    loss_stn = stn_loss(mat_64x64)

    # Total loss
    total_loss = loss_corner + loss_stn

    return total_loss


def dense_fusion_net_loss(
    pred_offsets, pred_scores, target_offsets, mat_64x64
) -> torch.Tensor:
    """
    Calculate dense fusion network loss, which is sum of unsupervised loss and spatial
    transformation regularization loss.

    Args:
        pred_offsets (torch.Tensor): Predicted offsets tensor.
        pred_scores (torch.Tensor): Predicted scores tensor.
        target_offsets (torch.Tensor): Target offsets tensor.
        mat_64x64 (torch.Tensor): Transformation matrix tensor.

    Returns:
        torch.Tensor: Total unsupervised loss.
    """
    eps = 1e-16
    weight = 0.1
    device = pred_offsets.device  # Get the device of the input tensors
    L1 = nn.SmoothL1Loss(reduction="none")

    # Move the target offsets tensor to the same device as pred_offsets
    target_offsets = target_offsets.to(device)

    loss_offset = L1(pred_offsets, target_offsets)  # [B x pnts x 8 x 3]
    loss_offset = torch.mean(loss_offset, (2, 3))  # [B x pnts]

    # Move other tensors to the same device
    pred_scores = pred_scores.to(device)
    mat_64x64 = mat_64x64.to(device)

    loss_unsupervised = (loss_offset * pred_scores) - (
        weight * torch.log(pred_scores + eps)
    )
    loss_unsupervised = loss_unsupervised.mean()  # [1]

    # loss2 for regularization
    loss_stn = stn_loss(mat_64x64)

    total_loss = loss_unsupervised + loss_stn

    return total_loss


def get_corner_offsets_from_pc(box_corners, pc) -> torch.Tensor:
    """
    Calculate corner offsets for each point in the point cloud, given the corners of the box.

    Args:
        box_corners (torch.Tensor): Corners tensor.
        pc (torch.Tensor): Point Cloud tensor.

    Returns:
        torch.Tensor: Corner offsets tensor.
    """
    batch_size, num_corners, _ = box_corners.shape
    num_points = pc.shape[1]

    corner_offsets = torch.zeros((batch_size, num_points, num_corners, 3))

    for b in range(0, batch_size):
        for n in range(0, num_points):
            for c in range(0, num_corners):
                corner_offsets[b, n, c, :] = pc[b, n, :] - box_corners[b, c, :]

    return corner_offsets


def get_corners_from_pred_offsets(
    pred_offsets: torch.Tensor, pred_scores: torch.Tensor
) -> torch.Tensor:
    """
    Calculate corners of the bounding box from the predicted offsets and scores.

    Args:
        pred_offsets (torch.Tensor): Predicted offsets tensor.
        pred_scores (torch.Tensor): Predicted scores tensor.

    Returns:
        torch.Tensor: Corners tensor.
    """
    batch_size, num_points, _, _ = pred_offsets.shape
    corners = torch.zeros((batch_size, 8, 3))

    max_confidence_indices = torch.argmax(pred_scores, dim=1)

    for b in range(0, batch_size):
        corners[b, :, :] = pred_offsets[b, max_confidence_indices[b], :, :]

    return corners
