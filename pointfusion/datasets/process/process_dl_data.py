import os
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

from pointfusion.utils.filepaths_utils import FilePaths
from pointfusion.utils.conversion_utils import (
    get_np_rgb_images,
    get_np_masks,
    get_np_point_clouds,
    get_np_bboxes,
)


class ProcessData:
    """
    Preprocess the raw dataset.

    It includes extracting the single object image patch of standard window size from the given RGB images,
    using the given masks. Also it extracts the point cloud data corresponding to the extracted object image
    patch.
    """

    def __init__(
        self,
        root: str,
        window_size: int,
        output_dir_path: str = None,
    ):
        """
        Creates an instance of ProcessData

        Args:
            data_root_dir (str): Path to the root directory of the raw dataset.
            window_size (int): Size of the window around the object.
            output_dir_path (str): Path to the output directory.
        """
        self.root_path = root
        self._file_paths = FilePaths(self.root_path)
        self.window_size = window_size
        self.output_dir = output_dir_path

        self._rgb_imgs_file_paths = self._file_paths.get_rgb_file_paths()
        self._masks_file_paths = self._file_paths.get_mask_file_paths()
        self._pc_file_paths = self._file_paths.get_pc_file_paths()
        self._bbox_file_paths = self._file_paths.get_bbox_file_paths()

        self._process_extracting()

    def _extract_objects_data_using_masks(
        self,
        rgb_img_path: str,
        masks_path: str,
        pc_path: str,
        bboxes_path: str,
    ):
        """
        Extracts the object images from the given RGB image using the given mask.

        Args:
            rgb_img_path (str): Path to the RGB image.
            masks_path (str): Path to the mask images.
            pc_path (str): Path to the point cloud data.

        Returns:
            List[np.ndarray]: List of cropped object images.
        """
        rgb_img = get_np_rgb_images(rgb_img_path)  # [H, W, 3]
        masks = get_np_masks(masks_path)  # [objects, H, W]
        pc = get_np_point_clouds(pc_path)  # [3, H, W]
        bboxes = get_np_bboxes(bboxes_path)  # [objects, 8, 3]

        # Folder name to stored extracted point clouds, object images and bounding boxes
        obj_rgb_folder_name = "object_images"
        obj_pc_folder_name = "object_point_clouds"
        obj_bbox_folder_name = "object_bboxes"

        # For each object in the image
        for i in range(masks.shape[0]):
            # Ensure the subdirectories exist or create them
            dir_name = os.path.basename(os.path.dirname(pc_path))
            dir_path = os.path.join(self.output_dir, dir_name)
            object_pc_dir_path = os.path.join(dir_path, obj_pc_folder_name)
            object_rgb_dir_path = os.path.join(dir_path, obj_rgb_folder_name)
            object_bbox_dir_path = os.path.join(dir_path, obj_bbox_folder_name)

            os.makedirs(dir_path, exist_ok=True)
            os.makedirs(object_pc_dir_path, exist_ok=True)
            os.makedirs(object_rgb_dir_path, exist_ok=True)
            os.makedirs(object_bbox_dir_path, exist_ok=True)

            binary_mask = masks[i, :, :]  # [H, W] (True/False)
            # Set pixels outside the mask to zero
            object_pixels = rgb_img * binary_mask[:, :, np.newaxis]
            object_pc = pc * binary_mask[np.newaxis, :, :]

            # Find the bounding box of the object
            indexes = np.argwhere(binary_mask)
            min_row, min_col = np.min(indexes, axis=0)
            max_row, max_col = np.max(indexes, axis=0)

            # Calculate the center of the bounding box
            center_row, center_col = (min_row + max_row) // 2, (min_col + max_col) // 2

            # Calculate the window boundaries
            window_min_row = max(0, center_row - self.window_size // 2)
            window_max_row = min(rgb_img.shape[0], center_row + self.window_size // 2)
            window_min_col = max(0, center_col - self.window_size // 2)
            window_max_col = min(rgb_img.shape[1], center_col + self.window_size // 2)

            # Extract the window around the object
            object_pixels = object_pixels[
                window_min_row:window_max_row, window_min_col:window_max_col, :
            ]
            object_pc = object_pc[
                :, window_min_row:window_max_row, window_min_col:window_max_col
            ]

            # Pad the image if size is not 224x224
            # Check if the size is exactly 224x224
            if (
                object_pixels.shape[0] != self.window_size
                or object_pixels.shape[1] != self.window_size
            ):
                # If not, calculate padding needed
                pad_rows = max(0, self.window_size - object_pixels.shape[0])
                pad_cols = max(0, self.window_size - object_pixels.shape[1])

                # Pad object_pixels to match the size of object_pc
                object_pixels = np.pad(
                    object_pixels,
                    ((0, pad_rows), (0, pad_cols), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                object_pc = np.pad(
                    object_pc,
                    (
                        (0, 0),
                        (0, self.window_size - object_pc.shape[1]),
                        (0, self.window_size - object_pc.shape[2]),
                    ),
                    mode="constant",
                    constant_values=0,
                )

            # Save the bounding box to the specified directory
            bbox_filename = f"object_{i+1}_3dbbox.npy"
            bbox_path = os.path.join(
                self.output_dir, dir_name, obj_bbox_folder_name, bbox_filename
            )
            np.save(bbox_path, bboxes[i, :, :])

            # Save the extracted point cloud to the specified directory
            point_cloud_filename = f"object_{i+1}.npy"
            point_cloud_path = os.path.join(
                self.output_dir, dir_name, obj_pc_folder_name, point_cloud_filename
            )
            np.save(point_cloud_path, object_pc)

            # Save the extracted object RGB image to the specified directory
            object_rgb_image_filename = f"object_{i+1}.jpg"
            object_rgb_image_path = os.path.join(
                self.output_dir,
                dir_name,
                obj_rgb_folder_name,
                object_rgb_image_filename,
            )
            cv2.imwrite(
                object_rgb_image_path, cv2.cvtColor(object_pixels, cv2.COLOR_RGB2BGR)
            )

    def _process_extracting(self):
        print(">>>>>>>>>>>>>>>>>>>>>> INFO <<<<<<<<<<<<<<<<<<<<<<\n")
        print(
            "Processing of given DL challenge datasets starts. Extracting object rgb images, point clouds, and bboxes using given masks."
        )
        print("Processed Data directory Structure:\n")
        print(f"root: {self.output_dir}")
        print("├── {sub_dir1}")
        print(f"│   ├── object_images")
        print(f"│   │   ├── object_1.jpg")
        print(f"│   │   └── object_2.jpg")
        print(f"│   ├── object_point_clouds")
        print(f"│   │   ├── object_1.npy")
        print(f"│   │   └── object_2.npy")
        print(f"│   ├── object_bboxes")
        print(f"│   │   ├── object_1_3dbbox.npy")
        print(f"│   │   └── object_2_3dbbox.npy")
        print("├── {sub_dir2}")
        print(">>>>>>>>>>>>>>>>>>>>>>       <<<<<<<<<<<<<<<<<<<<<<")

        total_files = len(self._pc_file_paths)
        for pc_path, bboxes_path, masks_path, rgb_img_path in tqdm(
            zip(
                self._pc_file_paths,
                self._bbox_file_paths,
                self._masks_file_paths,
                self._rgb_imgs_file_paths,
            ),
            total=total_files,
            desc="Processing DL Challenge Raw Data",
        ):
            self._extract_objects_data_using_masks(
                rgb_img_path, masks_path, pc_path, bboxes_path
            )
