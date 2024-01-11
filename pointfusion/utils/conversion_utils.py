from typing import Union, List

import numpy as np
import torch
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


def get_np_rgb_images(file_paths: Union[str, List[str]]) -> np.ndarray:
    """Get the RGB image(s) as a numpy array [H, W, 3].

    Args:
        file_paths (Union[str, List[str]]): File path(s) of the RGB image(s).

    Returns:
        image(s) (Union[np.ndarray, List[np.ndarray]]): List of numpy arrays representing the RGB image(s).

    Raises:
        ValueError: If the file_paths argument is not a string or a list.
    """
    if isinstance(file_paths, str):
        # Single file path
        return cv2.cvtColor(cv2.imread(file_paths), cv2.COLOR_BGR2RGB)  # [H, W, 3]
    elif isinstance(file_paths, list):
        # List of file paths
        images = []
        for path in file_paths:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            images.append(image)
        return images
    else:
        raise ValueError("Invalid file_paths argument. Expected a string or a list.")


def get_np_point_clouds(file_paths: Union[str, List[str]]) -> np.ndarray:
    """Get the point cloud as a numpy array [3, H, W].

    Args:
        file_paths (Union[str, List[str]]): File path(s) of the point cloud(s).

    Returns:
        point_clouds (Union[np.ndarray, List[np.ndarray]]): List of numpy arrays representing the point cloud(s).

    Raises:
        ValueError: If the file_paths argument is not a string or a list.
    """
    if isinstance(file_paths, str):
        # Single file path
        return np.load(file_paths)  # [3, H, W]
    elif isinstance(file_paths, list):
        # List of file paths
        point_clouds = []
        for path in file_paths:
            point_cloud = np.load(path)
            point_clouds.append(point_cloud)
        return point_clouds
    else:
        raise ValueError("Invalid file_paths argument. Expected a string or a list.")


def get_np_masks(file_paths: Union[str, List[str]]) -> np.ndarray:
    """Get the mask as a numpy array [objs, H, W].

    Args:
        file_paths (Union[str, List[str]]): File path(s) of the mask(s).

    Returns:
        masks (Union[np.ndarray, List[np.ndarray]]): List of numpy arrays representing the mask(s).

    Raises:
        ValueError: If the file_paths argument is not a string or a list.
    """
    if isinstance(file_paths, str):
        # Single file path
        return np.load(file_paths)  # [objs, H, W]
    elif isinstance(file_paths, list):
        # List of file paths
        masks = []
        for path in file_paths:
            mask = np.load(path)
            masks.append(mask)
        return masks
    else:
        raise ValueError("Invalid file_paths argument. Expected a string or a list.")


def get_np_bboxes(file_paths: Union[str, List[str]]) -> np.ndarray:
    """Get the bounding box as a numpy array [objs, 8, 3].

    Args:
        file_paths (Union[str, List[str]]): File path(s) of the bounding box(es).

    Returns:
        bboxes (Union[np.ndarray, List[np.ndarray]]): List of numpy arrays representing the bounding box(es).

    Raises:
        ValueError: If the file_paths argument is not a string or a list.
    """
    if isinstance(file_paths, str):
        # Single file path
        return np.load(file_paths)  # [objs, 8, 3]
    elif isinstance(file_paths, list):
        # List of file paths
        bboxes = []
        for path in file_paths:
            bbox = np.load(path)
            bboxes.append(bbox)
        return bboxes
    else:
        raise ValueError("Invalid file_paths argument. Expected a string or a list.")
