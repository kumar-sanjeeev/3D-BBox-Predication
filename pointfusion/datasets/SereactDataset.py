import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from pointfusion.utils.vis_utils import visualize_rgb_image, visualize_point_cloud
from pointfusion.utils.draw_utils import draw_bounding_boxes


class SereactDataset(Dataset):
    """
    This class provides a dataset in form of object rgb image, its corresponding
    3D point clouds, and GT bounding box for training and testing.
    """

    def __init__(self, root_path: str):
        """
        Creates an instance of SereactDataset.

        Args:
            root_path (str): Root path of the processed dataset folder.
        """
        if not root_path:
            raise ValueError("Please provide the root path to the raw dataset")

        if not os.path.isabs(root_path):
            raise ValueError("`root_path` and  must be absolute paths.")

        self.root_path = root_path
        self._data_paths = self._load_data_paths()

    def _load_data_paths(self):
        """
        Load data paths from the processed data folder.

        Returns:
            List: A list of tuples containing image path, point cloud path, and bounding box path for each object.

        {$output_dir}                   (root)
        ├── sub_dir1                    (for ex. 8b061a8a-9915-11ee-9103-bbb8eae05561 )
        │   ├── object_images
        │   │   ├── object_1.jpg
        │   │   └── object_2.jpg
        │   ├── object_point_clouds
        │   │   ├── object_1.npy
        │   │   └── object_2.npy
        │   ├── object_bboxes
        │   │   ├── object_1_3dbbox.npy
        │   │   └── object_2_3dbbox.npy
        ├── sub_dir_2
        └── ....
        """
        output_data_paths = []
        sub_dirs = []

        for sub_dir in os.listdir(self.root_path):
            sub_dir_path = os.path.join(self.root_path, sub_dir)
            if os.path.isdir(sub_dir_path):
                sub_dirs.append(sub_dir_path)

        for sub_dir in sub_dirs:
            object_images_path = os.path.join(sub_dir, "object_images")
            object_point_clouds_path = os.path.join(sub_dir, "object_point_clouds")
            object_bboxes_path = os.path.join(sub_dir, "object_bboxes")

            image_files = [
                f for f in os.listdir(object_images_path) if f.endswith(".jpg")
            ]

            for image_file in image_files:
                image_file_path = os.path.join(object_images_path, image_file)
                point_cloud_file_path = os.path.join(
                    object_point_clouds_path, image_file.replace(".jpg", ".npy")
                )
                bbox_file_path = os.path.join(
                    object_bboxes_path, image_file.replace(".jpg", "_3dbbox.npy")
                )

                if (
                    os.path.exists(image_file_path)
                    and os.path.exists(point_cloud_file_path)
                    and os.path.exists(bbox_file_path)
                ):
                    output_data_paths.append(
                        (image_file_path, point_cloud_file_path, bbox_file_path)
                    )

        return output_data_paths

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self._data_paths)

    def __getitem__(self, idx):
        """
        Gets the item at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: A dictionary containing RGB image, point cloud, and bounding box for the specified index.
        """
        object_image_path, object_point_cloud_path, object_bbox_path = self._data_paths[
            idx
        ]

        # Read Image
        rgb_img_np = cv2.imread(object_image_path)
        rgb_img_torch = torch.tensor(rgb_img_np, dtype=torch.float32).permute(2, 0, 1)

        # Read Point Cloud
        pc_np = np.load(object_point_cloud_path)
        pc_torch = torch.tensor(pc_np, dtype=torch.float32).view(3, -1)

        # Read BBox
        bbox_np = np.load(object_bbox_path)
        bbox_torch = torch.tensor(bbox_np, dtype=torch.float32)

        return {
            "img_np": rgb_img_np,
            "pc_np": pc_np,
            "bbox_np": bbox_np,
            "img_t": rgb_img_torch,
            "pc_t": pc_torch,
            "bbox_t": bbox_torch,
        }

    def visualize_data_at_idx(self, idx):
        """
        Visualizes the RGB image, point cloud, and bounding box for the specified index.

        Args:
            idx (int): Index of the item to visualize.
        """
        sample = self[idx]
        image = sample["img_np"]
        pc = sample["pc_np"]
        bbox = sample["bbox_np"]

        visualize_rgb_image(image)
        visualize_point_cloud(pc)
        draw_bounding_boxes(pc, bbox)
