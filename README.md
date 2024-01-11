# 3D-BBox-Predication
This repository contains the solution in form of python package named `pointfusion` to solve the DL challenge.

## Project Structure Overview
The directory structure of this project looks like this:

```.
├── configs                                      <- Hydra Configs
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
│   ├── input_data_vis.py                            <- Python file to visualization
│   ├── run_process_data.py                          <- Python file to create dataset after preprocessing
│   ├── trainer.py                                   <- Python file to train/validate/test the model
├── README.md
├── requirements.txt
└── setup.py
```

## QuickStart
```bash
# clone this repo
git https://github.com/kumar-sanjeeev/3D-BBox-Predication.git
cd 3D-BBox-Predication

# create and activate python virtual env
python -m venv .venv .
source .venv/bin/activate  

# install requirements
pip install -r requirements.txt

# locally install the poinfusion package
pip install -e .
```

## Setting the environment variables
To use the code from this repository, following environment variables need to be set. Put these into your `~/.bashrc file`. The reason for this one is avoid hard-coding data path into the codebase.

```bash
export SEREACT_DATA_PATH=</path/to/downloaded/dl_challenge_dataset> # for eg. SEREACT_DATA_PATH=/home/user/3D-BBox-Predication/dl_challenge_processed

export SEREACT_PROCESSED_DATA_PATH=<path/to/store/processed_dl_challenge_dataset> # for eg. SEREACT_PROCESSED_DATA_PATH=/home/user/3D-BBox-Predication/dl_challenge_processed
```

## How to run the 3D Bounding Box Predication Pipeline

**Step1 :** Download the dl challenge dataset into the local system from the provided link

**Step2 :** Set the env variable `SEREACT_DATA_PATH` to path where dataset is downloaded

**Step3 :** Set the env varible `SEREACT_PROCESSED_DATA_PATH` to a path where you want to store the processed dataset.

**Step4 :** Run the following file to start preprocessing the data.
```bash
# assuming you're inside the `3D-BBox-Predication`
python3 pointfusion/run_process_data.py 
```
**Step5 :** Start the training of model as follows:
```bash
# assuming you're inside the `3D-BBox-Predication`
python3 pointfusion/trainer.py
```