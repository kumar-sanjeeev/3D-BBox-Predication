import hydra
import numpy as np

from pointfusion.utils.filepaths_utils import FilePaths
from pointfusion.utils.vis_utils import (
    visualize_rgb_image,
    visualize_point_cloud,
    visualize_rgb_and_masks,
)
from pointfusion.utils.draw_utils import draw_bounding_boxes


@hydra.main(version_base=None, config_path="../configs", config_name="vis")
def main(cfg):
    # Load the given data
    filepaths = FilePaths(root_path=cfg.raw_data_dir_path)

    # Load the files
    rgb_file_paths = filepaths.get_rgb_file_paths()
    pc_file_paths = filepaths.get_pc_file_paths()
    mask_file_paths = filepaths.get_mask_file_paths()
    bbox_file_paths = filepaths.get_bbox_file_paths()

    # Visualize the RGB image of give sample
    visualize_rgb_image(rgb_file_paths[cfg.sample])

    # Visualize the RGB image and its masks of given sample
    visualize_rgb_and_masks(rgb_file_paths[cfg.sample], mask_file_paths[cfg.sample])

    # Visualize the point cloud of the given sample
    visualize_point_cloud(pc_file_paths[cfg.sample])

    # Draw the bounding boxes on the Point Cloud of the given sample
    draw_bounding_boxes(pc_file_paths[cfg.sample], bbox_file_paths[cfg.sample])


if __name__ == "__main__":
    main()
