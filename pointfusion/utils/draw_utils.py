from typing import Union

import open3d as o3d
import numpy as np


def draw_bounding_boxes(
    pc_input: Union[str, np.ndarray], bboxes_input: Union[str, np.ndarray]
) -> None:
    """
    Draw 3D bounding boxes on a point cloud using Open3D.

    Args:
        pc_input (str or numpy.ndarray): Point cloud as a numpy array of shape (3, H, W) or file path.
        bboxes_input (str or numpy.ndarray): Bounding boxes as a numpy array of shape (objects, 8, 3) or file path.
    """

    if (not isinstance(pc_input, (str, np.ndarray))) or (
        not isinstance(bboxes_input, (str, np.ndarray))
    ):
        raise ValueError(
            "Invalid input type. Expected a file path (str) or a NumPy array."
        )

    # Load the point_cloud is input is a file path
    if isinstance(pc_input, str):
        pc = np.load(pc_input)
    else:
        pc = pc_input

    # Load the bounding boxes if input is a file path
    if isinstance(bboxes_input, str):
        bboxes = np.load(bboxes_input)
    else:
        bboxes = bboxes_input

    # Check if bounding box is a single box of shape [8,3]
    if len(bboxes.shape) == 2:
        bboxes = np.expand_dims(bboxes, axis=0)

    # Reshape the point cloud to (H*W)x3
    if len(pc.shape) == 3:
        pc = pc.transpose(1, 2, 0).reshape(-1, 3)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)

    all_lines = []
    all_colors = []

    for i, vertices in enumerate(bboxes):
        lines = [
            [0, 1],  # Bottom edges
            [1, 2],
            [2, 3],
            [3, 0],  # Top edges
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Vertical edges
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        colors = [
            [1, 0, 0] for _ in range(len(lines))
        ]  # Red color for the bounding box edges

        # Shift indices for the new object
        lines = [[idx + i * 8 for idx in line] for line in lines]

        all_lines.extend(lines)
        all_colors.extend(colors)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.concatenate(bboxes, axis=0))
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.visualization.draw_geometries([point_cloud, line_set])
