# 3D-BBox-Prediction

This repository contains a Python package named `pointfusion` to solve a DL challenge related to 3D bounding box prediction.

## Table of Contents
- [Project Structure Overview](#project-structure-overview)
- [Quick Start](#quick-start)
- [How to Run the 3D Bounding Box Prediction Pipeline](#how-to-run-the-3d-bounding-box-prediction-pipeline)
- [About Configuration Files](#about-configuration-files)

## Project Structure Overview

```.
├── configs                                      <- Hydra Configs
│   ├── preprocess.yaml                             <- Preprocessing Config 
│   ├── train.yaml                                  <- Training Config 
│   └── vis.yaml                                    <- Data Visualization Config
├── data                                         <- Data
│   └── dl_challenge                                <- Given dl challenge data
├── pointfusion                                  <- PointFusion Python Package
│   ├── datasets                                 <- Dataset
│   │   ├── __init__.py
│   │   ├── process
│   │   │   └── process_dl_data.py                  <- To pre-process the dataset
│   │   └── SereactDataset.py                       <- To create Pytorch Dataset
│   ├── losses                                   <- Losses
│   │   ├── __init__.py
│   │   └── loss.py                                  <- To define different losses
│   ├── models                                   <- DL Models
│   │   ├── __init__.py
│   │   └── pointfusion.py                           <- Main PointFusion Model
│   ├── modules
│   │   ├── pointnet.py                              <- PointNet
│   │   ├── resnet.py                                <- Pretrained ResNet 
│   │   └── tnet.py                                  <- Spatial Transformation Net
│   └── utils
│   │   │
│   │   ├── conversion_utils.py                      <- Data Conversion utils
│   │   ├── draw_utils.py                            <- Drawing utils
│   │   ├── filepaths_utils.py                       <- FilePaths utils
│   │   ├── __init__.py
│   │   ├── iou_utils.py                             <- Evaluation Metric utils
│   │   ├── objectron                     
│   │   │   ├── box.py
│   │   │   ├── __init__.py
│   │   │   └── iou.py
│   │   └── vis_utils.py                             <- visualization utils   
│   ├── outputs                                  <- Hydra Outputs
│   ├── lightning_logs                           <- PyTorch Lightning logs
|   |
│   ├── __init__.py
│   ├── input_data_vis.py                            <- Python file to visualize the input raw data
│   ├── run_process_data.py                          <- Python file to create dataset after preprocessing
│   ├── trainer.py                                   <- Python file to train/validate/test the model
├── README.md
├── requirements.txt
└── setup.py
```


## Quick Start
1. **Clone this repository:**
    ```bash
    git clone https://github.com/kumar-sanjeeev/3D-BBox-Predication.git
    cd 3D-BBox-Predication
    ```

2. **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install the `pointfusion` package locally:**
    ```bash
    pip install -e .
    ```


## How to Run the 3D Bounding Box Prediction Pipeline

**Step1 :** Download the dl challenge dataset into the local system from the provided link.

**Step2 :** Set the `root_path` parameter in `config/preprocess.yaml` and `config/vis.yaml` file to absolute path where dl challenge dataset is downloaded.

**Step3 :** Set the `processed_data_path` parameter in `config/preprocess.yaml` and `config/train.yaml` file to absolute path where you want to store the preprocessed data.

**Step4 :** Run the data preprocessing.
```bash
# assuming you're inside the `3D-BBox-Predication`
python3 pointfusion/run_process_data.py 
```
**Step5 :** START TRAINING MODEL.
```bash
# assuming you're inside the `3D-BBox-Predication`
python3 pointfusion/trainer.py
```
**Step6 :** Visualize training, validation, and test metrics on Tensorboard:

```bash
# assuming you're inside `3D-BBox-Predication`
tensorboard --logdir=pointfusion/lightning_logs/
```

## About Configuration Files

The project uses yaml based configuration files to manage various settings for preprocessing, training, and visualization. The default configuration files are located in the `configs` directory:

- `preprocess.yaml`: Configuration for data preprocessing.
- `train.yaml`: Configuration for training the model.
- `vis.yaml`: Configuration for data visualization.

### Default Configuration Files

In the `input_data_vis.py` script, the default configuration files used are:

- `configs/vis.yaml`

In the `run_process_data.py` script, the default configuration files used are:

- `configs/preprocess.yaml`


In the `trainer.py` script, the default configuration files used are:


- `configs/train.yaml`

### Modifying Configuration Files

To customize the behavior of the preprocessing, training, or visualization, you can edit the corresponding configuration files. Each file contains key-value pairs that define various parameters such as batch size, learning rate, model architecture, etc.

For example, to change the batch size for training, open `configs/train.yaml` and locate the `batch_size` parameter. Modify the value according to your preference.

### Running Trainer with Custom Configuration

If you want to use a custom configuration file, you can specify it when running the `trainer.py` script. For example:

```bash
# for example
python3 pointfusion/trainer.py --config_path=configs/custom_train_config.yaml
