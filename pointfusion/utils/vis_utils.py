from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from pointfusion.utils.conversion_utils import (
    get_np_rgb_images,
    get_np_point_clouds,
    get_np_masks,
)


FIGURE_WIDTH = 10
FIGURE_HEIGHT = 8


def visualize_rgb_image(rgb_input: Union[str, np.ndarray]) -> None:
    """Visualize the RGB Image.

    Args:
        rgb_input (str or numpy.ndarray): File path or numpy array of the point cloud.

    Returns:
        None

    Raises:
        ValueError: If rgb_input is neither a valid file path nor a numpy array.
    """

    # Load the RGB image if the input is a file path
    if isinstance(rgb_input, str):
        rgb_image = get_np_rgb_images(rgb_input)
    # Load the RGB image if the input is a numpy array
    elif isinstance(rgb_input, np.ndarray):
        rgb_image = rgb_input
    else:
        raise ValueError(
            "Invalid rgb_input argument. Expected a file path or a numpy array."
        )

    plt.imshow(rgb_image)
    plt.show()


def visualize_rgb_and_masks(
    rgb_input: Union[str, np.ndarray], masks_input: Union[str, np.ndarray]
) -> None:
    """Visualize the RGB image and its masks together.

    Args:
        rgb_input (str or np.ndarray): File path or NumPy array of the RGB image.
        masks_input (str or np.ndarray): File path or NumPy array of the masks corresponding to the RGB image.

    Returns:
        None

    Raises:
        ValueError: If rgb_input or masks_input is an invalid type or format.
    """

    if (not isinstance(rgb_input, (str, np.ndarray))) or (
        not isinstance(masks_input, (str, np.ndarray))
    ):
        raise ValueError(
            "Invalid input type. Expected a file path (str) or a NumPy array."
        )

    if isinstance(rgb_input, str):
        # Load RGB image if the input is a file path
        rgb_image = get_np_rgb_images(rgb_input)
    else:
        rgb_image = rgb_input

    if isinstance(masks_input, str):
        # Load masks if the input is a file path
        mask_images = get_np_masks(masks_input)
    else:
        mask_images = masks_input

    num_masks, h, w = mask_images.shape

    # Create a grid of subplots
    rows = int(np.ceil(np.sqrt(num_masks + 1)))
    cols = int(np.ceil((num_masks + 1) / rows))

    # Create a new figure and set a title
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.suptitle("RGB and Masks", fontsize=16)

    # Plot RGB image
    plt.subplot(rows, cols, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis("off")

    # Plot each mask in a subplot
    for i in range(num_masks):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(mask_images[i], cmap="viridis", vmin=0, vmax=1)
        plt.title(f"Mask {i + 1}")
        plt.axis("off")

    plt.show()


def visualize_point_cloud(pc_input: Union[str, np.ndarray]) -> None:
    """Visualize the point cloud.

    Args:
        pc_input (str or numpy.ndarray): File path or numpy array of the point cloud.

    Returns:
        None

    Raises:
        ValueError: If pc_input is neither a valid file path nor a numpy array.
    """

    if isinstance(pc_input, str):  # If input is a file path
        point_cloud = get_np_point_clouds(pc_input)
    elif isinstance(pc_input, np.ndarray):  # If input is a numpy array
        point_cloud = pc_input
    else:
        raise ValueError(
            "Invalid input_data argument. Expected a file path or a numpy array."
        )

    # Reshape the point cloud if necessary
    if len(point_cloud.shape) == 3:
        point_cloud = point_cloud.transpose(1, 2, 0).reshape(-1, 3)

    # Create an Open3D PointCloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pc])
