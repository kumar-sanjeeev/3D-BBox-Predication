import numpy as np

import pointfusion.utils.objectron.box as Box


def np_get_3d_bounding_box_including_center(corners):
    # Calculate the min and max coordinates for each axis
    x_min = np.min(corners[:, 0])
    x_max = np.max(corners[:, 0])

    y_min = np.min(corners[:, 1])
    y_max = np.max(corners[:, 1])

    z_min = np.min(corners[:, 2])
    z_max = np.max(corners[:, 2])

    center = np.array(
        [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    ).reshape(1, -1)

    # Append the calculated center as the first element
    corners = np.concatenate((center, corners))
    box = Box.Box(corners)
    return box
