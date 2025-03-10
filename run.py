import torch
from torchvision import models, transforms
import qai_hub
import pandas as pd
import numpy as np
import os
import subprocess
from PIL import Image
from typing import Tuple

# Load key.csv (Ground Truth Labels)
key_csv_path = "key.csv"  # Update this path if necessary
key_df = pd.read_csv(key_csv_path)
ground_truth = dict(zip(key_df["file_name"], key_df["class_index"]))
print(ground_truth)
file_names = key_df["file_name"].tolist()

# Custom wrapper class for preprocessing and MobileNetV2
class PreprocessedMobileNetV2(torch.nn.Module):
    def __init__(self, num_classes, pretrained_weights_path):
        super(PreprocessedMobileNetV2, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
        state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cpu'))
        self.mobilenet_v2.load_state_dict(state_dict)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, img):
        return self.mobilenet_v2(img)

# Step 1: Upload Dataset
try:
    # result = subprocess.run(["python", "upload_dataset.py"], capture_output=True, text=True)
    # dataset_id = result.stdout.strip()  # Assuming dataset_id is printed by upload_dataset.py
    dataset_id = "dv74qmp02"
    print(f"Dataset ID: {dataset_id}")

    if not dataset_id:
        raise ValueError("Dataset ID is empty. Check upload_dataset.py output.")
except Exception as e:
    print(f"Error running upload_dataset.py: {str(e)}")
    exit(1)

# Step 2: Load Model
num_classes = 64
pretrained_path = "./model/mobilenet_v2_coco.pth"  # Update path
model = PreprocessedMobileNetV2(num_classes=num_classes, pretrained_weights_path=pretrained_path)
model.eval()

# Step 3: Trace Model
input_shape: Tuple[int, ...] = (1, 3, 224, 224)
example_input = torch.rand(input_shape)
pt_model = torch.jit.trace(model, example_input)

# Step 4: Compile Model
compile_job = qai_hub.submit_compile_job(
    pt_model,
    name="coco_imagenet",
    device=qai_hub.Device("Samsung Galaxy S24 (Family)"),
    input_specs=dict(image=input_shape),
)
compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])

# Retrieve Compiled Model
compiled_model = compile_job.get_target_model()
print(f"Compiled Model: {compiled_model}")

if compiled_model is None:
    raise ValueError("Compiled model is None. Check if compilation succeeded.")



device_name = "Samsung Galaxy S24 (Family)"

device = qai_hub.Device(device_name)

# Step 7: Load and Preprocess Dataset Images
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

try:
    # Step 1: Submit Inference Job
    inference_output = qai_hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=qai_hub.get_dataset(dataset_id),
        options="--max_profiler_iterations 1"
    ).download_output_data()

    # Step 2: Extract Model Output
    output_logits = inference_output["output_0"]  # Assuming model output key is 'output_0'

    # Convert to NumPy array for processing
    predicted_classes = np.array(output_logits)
    print("Predicted classes raw shape:", predicted_classes.shape)

    # Step 3: Remove Extra Dimensions
    predicted_classes = np.squeeze(predicted_classes)  # Removes unnecessary dimensions
    print("Predicted classes shape after squeeze:", predicted_classes.shape)

    # Step 4: Ensure Shape is (100, 64)
    if predicted_classes.shape == (100, 64):  
        predicted_labels = np.argmax(predicted_classes, axis=1)
        print("Predicted class indices:", predicted_labels)
    else:
        print(f"Unexpected predicted_classes shape after squeeze: {predicted_classes.shape}")
        exit(1)

    # Step 5: Compute Accuracy
    correct_predictions = sum(
        int(predicted_labels[i]) == int(ground_truth[file_names[i]])
        for i in range(len(file_names))
    )

    accuracy = (correct_predictions / len(file_names)) * 100
    print(f"Model Accuracy: {accuracy:.2f}%")

    # Step 6: Save Accuracy to File
    with open("accuracy_log.txt", "w") as f:
        f.write(f"Model accuracy: {accuracy:.2f}%\n")

except Exception as e:
    print(f"Inference job failed: {str(e)}")
    exit(1)
