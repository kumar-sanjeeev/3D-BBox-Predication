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
git clone https://github.com/kumar-sanjeeev/3D-BBox-Predict.git
cd 3D-BBox-Predication

# create and activate python virtual env
python -m venv .3db_env .
source .3db_env/bin/activate  

# install requirements
pip install -r requirements.txt
```