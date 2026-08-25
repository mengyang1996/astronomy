# MBP CenterNet Detection Project

This project contains two Python scripts designed to take images provided by Subomi, train a machine learning model to detect point sources (MBPs), and visually test the model's accuracy on unseen data.

The system uses a Simple U-Net architecture to generate heatmaps, predicting the precise center of each target object instead of drawing bounding boxes around them.

## Overview of the Workflow

The workflow is broken into two main steps:

1. **Training:** The model learns how to find MBPs by looking at training images and their corresponding labels.


2. **Testing/Evaluation:** The trained model is tested on brand new images to see how well it performs, generating visual comparisons.



---

## 1. Training the Model

The first script is responsible for teaching the model how to identify targets.

### How it Works

* **Data Processing:** It reads grayscale images and resizes them to 640x640 pixels.


* **Label Conversion:** It takes standard YOLO-formatted bounding box labels and converts them into centroid heatmaps.


* **Heatmap Generation:** These heatmaps use a Gaussian radius to highlight the exact center of the target.


* **Training Loop:** The U-Net model trains over 100 epochs using batches of 8 images at a time.


* **Loss Calculation:** It uses a custom `HeatmapFocalLoss` function to penalize incorrect predictions.



### Outputs

* The script tracks the average loss during training.


* It automatically saves the weights of the most recent model as `last_centernet.pt`.


* It saves the best-performing model as `best_centernet.pt`.



---

## 2. Testing the Model

The second script evaluates the U-Net model using the `best_centernet.pt` weights generated during training.

### How it Works

* **Image Selection:** It reads a file named `results.txt` to find unseen images.


* **Random Sampling:** It randomly selects 50 unseen images from that list to test the model.


* **Prediction:** The script feeds these images into the trained model to generate predicted heatmaps.


* **Peak Extraction:** It extracts specific coordinate points from the heatmaps using a minimum confidence threshold of 0.3.



### Outputs

* The script generates a 2-panel side-by-side plot for each tested image.


* The left panel displays the **Ground Truth**, marking the actual MBPs with green circles.


* The right panel displays the **Prediction**, marking what the model found with red crosses.


* These comparison plots are automatically saved as `.png` files in the model's directory for easy review.



---

## Setup & Requirements

To ensure the scripts run properly, make sure your data folders are set up to match the paths in the code:

* Your training images must be in `.jpg` or `.JPG` format.


* Your labels must be in `.txt` format and contain class `1` for valid MBPs.


* Ensure both scripts have access to a CUDA-enabled GPU for faster processing, though they will default to CPU if a GPU is unavailable.
