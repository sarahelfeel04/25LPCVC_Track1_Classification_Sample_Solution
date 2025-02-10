# LPCVC 2025 Track 1 - Image classification for different lighting conditions and styles

## Overview

This repository contains Python scripts designed to manage, compile, evaluate, and submit machine learning models using the `qai_hub` library.



## **Table of Contents**

1.  [Features](#features)
2.  [Installation](#installation)
3.  [Usage](#usage)
4.  [Requirements](#requirements)

----------

## **Features**

-   **Pretrained MobileNetV2**: Load MobileNetV2 with custom classes and pretrained weights.
-   **Preprocessing Pipeline**: Includes resizing and normalization for inputs.
-   **Model Compilation**: Supports compiling the model for a specific target device using QAI Hub.
-   **Inference Job Submission**: Submit and retrieve inference results via QAI Hub.

----------

## **Installation**

### **Step 1: Clone the Repository**

### **Step 2: Install Dependencies**

Ensure you have Python 3.9+ installed. Install the required Python packages:

`pip install -r requirements.txt` 

----------

## **Requirements**

-   Python 3.9+
-   Torch and torchvision
-   QAI Hub
-   Required packages listed in `requirements.txt`
----------

## **Usage**

### F

### **1. Prepare Pretrained Weights**

Place the pretrained MobileNetV2 weights (`.pth` file) in the `./model/` directory. Update the `pretrained_path` variable in the script to point to your weights file.

### **2. Run the Script**

Execute the script to perform model compilation and inference:

`python run.py` 

### **3. Modify Parameters**

-   **Upload Dataset**: Upload your own dataset or sample dataset to run inference jobs.
-   **Target Device**: Update the `device` parameter to specify your target device.
    

----------

## **Key Notes**

-   The script uses a preprocessing pipeline compatible with ImageNet-trained models.
-   Ensure the target device is supported by QAI Hub for successful model compilation.