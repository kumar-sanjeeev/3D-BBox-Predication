import os
from dataclasses import dataclass
from typing import List


@dataclass
class FilePaths:
    """
    To get the absolute file paths of different data files
    from the root path of the raw data folder.

    Attributes:
        root_path (str): Root path to the raw data folder.
    """

    root_path: str

    def _get_file_paths(self, prefix: str) -> List[str]:
        """Get the file paths of files with the given prefix.

        Args:
            prefix (str): Prefix of the files.

        Returns:
            List[str]: List of file paths.
        """
        paths = []
        for root, _, files in os.walk(self.root_path):
            for file in files:
                if file.startswith(prefix):
                    paths.append(os.path.join(root, file))
        return paths

    def get_pc_file_paths(self) -> List[str]:
        """Get the absoulte file paths of point cloud files.

        Returns:
            List[str]: List of file paths for point cloud files.
        """
        return self._get_file_paths("pc")

    def get_rgb_file_paths(self) -> List[str]:
        """Get the absolute file paths of RGB image files.

        Returns:
            List[str]: List of file paths for RGB image files.
        """
        return self._get_file_paths("rgb")

    def get_mask_file_paths(self) -> List[str]:
        """Get the absolute file paths of mask image files.

        Returns:
            List[str]: List of file paths for mask image files.
        """
        return self._get_file_paths("mask")

    def get_bbox_file_paths(self) -> List[str]:
        """Get the file absoulte paths of bounding box files.

        Returns:
            List[str]: List of file paths for bounding box files.
        """
        return self._get_file_paths("bbox3d")
