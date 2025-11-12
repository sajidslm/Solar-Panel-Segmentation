# Solar-Panel-Segmentation
This project focuses on YOLO-based segmentation of individual solar panels in aerial imagery. The dataset comprises 20 labeled, high-resolution RGB images, each containing between 150 and over 700 solar panels. Within a given solar farm, the panels are largely uniform in appearance, although variations may occur due to differences in lighting conditions and viewing angles.

## Dataset Preparation
The dataset contains annotations in COCO format, which are converted into .txt files to meet YOLO's labelling requirements. These YOLO-formatted labels are used in the training process. However they have been also used to generate annotated images for label verification (its an optional step).  Following this, the dataset is divided into training and validation subsets, with 60% allocated for training and the remaining 40% for validation. This split strategy accounts for the limited number of images in the dataset and aims to enhance the size of the validation set for more robust evaluation.

## Training and Validation
The model employed for training is yolo11n.pt, though it can be substituted with other YOLO variants as needed. The training configuration includes the following parameters: 150 epochs, an image size of 640 pixels, and a batch size of 4.

## Install Dependencies
Use the requirements.txt file to install all necessary libraries. You can do this by running:
pip install -r requirements.txt

## Update Dataset Path
Open the solar_train_valid.py script and locate the line:
im_folder = "F:/Image Datasets/ Solar Segmentation"  # Dataset directory path

## Run the Training Script
Once the dependencies are installed and the dataset path is set, execute the script:
python solar_train_valid.py

## Results
The results are provided in the train folder with fine tune models weights

