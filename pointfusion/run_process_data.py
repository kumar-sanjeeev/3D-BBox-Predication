from pointfusion.datasets.process.process_dl_data import ProcessData

if __name__ == "__main__":
    root = "/home/robotics-noob/Workspaces/3D_BBox_Predication_ws/data/dl_challenge_raw"
    window_size = 224
    output_dir_path = "/home/robotics-noob/Workspaces/3D_BBox_Predication_ws/data/dl_challenge_processed_224"
    pp = ProcessData(root, window_size, output_dir_path)
