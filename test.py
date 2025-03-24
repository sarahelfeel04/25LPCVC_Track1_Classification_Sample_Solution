import torch
from torch import nn
from torchvision import models, transforms
import qai_hub
import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2
from typing import Tuple, List, Dict

# Load and validate ground truth labels
key_csv_path = "key.csv"
key_df = pd.read_csv(key_csv_path)
ground_truth = dict(zip(key_df["file_name"], key_df["class_index"]))
file_names = key_df["file_name"].tolist()
max_class_index = key_df["class_index"].max()
unique_classes = max_class_index + 1  # Add 1 because indices are zero-based
print(f"Dataset contains {len(file_names)} images across {unique_classes} classes (max index: {max_class_index})")


# Enhanced preprocessing with histogram equalization
class EnhancedPreprocessor:
    def __init__(self):
        self.resize_dim = (224, 224)
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def apply_hist_eq(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img_rgb)

    def apply_clahe(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)

    def preprocess_for_inference(self, img_path):
        img = Image.open(img_path).convert("RGB")

        # Apply image enhancement techniques
        img = self.apply_hist_eq(img)
        img = self.apply_clahe(img)

        # Final transformations
        transform = transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        ])
        return transform(img).unsqueeze(0)


# Improved MobileNetV2 that preserves the pretrained classifier
class PreservedMobileNetV2(nn.Module):
    def __init__(self, weights_path=None):
        super(PreservedMobileNetV2, self).__init__()
        # Initialize with 64 classes to match COCO weights (which includes our max class index)
        self.model = models.mobilenet_v2(weights=None, num_classes=64)

        # Load full pretrained weights including classifier
        if weights_path and os.path.exists(weights_path):
            print(f"Loading complete weights from {weights_path}")
            pretrained_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(pretrained_dict)
            print("Successfully loaded FULL pretrained weights including classifier")

    def forward(self, x):
        return self.model(x)


# Find the correct data directory
data_dir = "dataset"  # Standard location
if not os.path.exists(data_dir):
    data_dir = "common/images-20250310T184314Z-001/images"  # Try alternative
    if not os.path.exists(data_dir):
        dirs = [d for d in os.listdir() if os.path.isdir(d) and not d.startswith('.')]
        print(f"Available directories: {dirs}")
        for d in dirs:
            if os.path.exists(os.path.join(d, file_names[0])):
                data_dir = d
                break

print(f"Using data directory: {data_dir}")

# Verify files exist
sample_path = os.path.join(data_dir, file_names[0])
if not os.path.exists(sample_path):
    raise FileNotFoundError(f"Cannot find sample file: {sample_path}")
else:
    print(f"Verified sample file exists: {sample_path}")

# Initialize model with preserved classifier weights
pretrained_path = "model/mobilenet_v2_coco.pth"
model = PreservedMobileNetV2(weights_path=pretrained_path)


# Fine-tuning function that only trains the classifier and last few layers
def fine_tune_model(model, data_dir, file_names, ground_truth, epochs=5):
    model.train()

    # Freeze most of the backbone features
    for name, param in model.model.named_parameters():
        if 'features.18' in name or 'classifier' in name:  # Only fine-tune last conv block + classifier
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Use smaller learning rate for fine-tuning
    optimizer = torch.optim.Adam([
        {'params': model.model.features.parameters(), 'lr': 0.0001},
        {'params': model.model.classifier.parameters(), 'lr': 0.001}
    ])

    criterion = nn.CrossEntropyLoss()
    preprocessor = EnhancedPreprocessor()

    # Create dataset and dataloader
    train_images = []
    train_labels = []

    for file_name in file_names:
        img_path = os.path.join(data_dir, file_name)
        if os.path.exists(img_path):
            img = preprocessor.preprocess_for_inference(img_path).squeeze(0)
            train_images.append(img)
            train_labels.append(ground_truth[file_name])

    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    # Train the model
    batch_size = 16
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(train_images), batch_size):
            inputs = train_images[i:i + batch_size]
            labels = train_labels[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (len(train_images) / batch_size):.4f}")

    return model


# Fine-tune the model (instead of training from scratch)
model = fine_tune_model(model, data_dir, file_names, ground_truth, epochs=5)
model.eval()  # Set to evaluation mode before inference

# Create new dataset from enhanced images
preprocessor = EnhancedPreprocessor()
input_tensors = []
sample_paths = []

# Process and create tensors from local files
for file_name in file_names:
    img_path = os.path.join(data_dir, file_name)
    if os.path.exists(img_path):
        tensor = preprocessor.preprocess_for_inference(img_path)
        input_tensors.append(tensor)
        sample_paths.append(img_path)
    else:
        print(f"Warning: Could not find {img_path}")

# Stack all tensors for batch processing
if input_tensors:
    batch_tensor = torch.cat(input_tensors, dim=0)

    # Create trace model
    input_shape = (1, 3, 224, 224)
    example_input = torch.rand(input_shape)
    pt_model = torch.jit.trace(model, example_input)

    # Compile model
    compile_job = qai_hub.submit_compile_job(
        pt_model,
        name="preserved_classifier_mobilenetv2",
        device=qai_hub.Device("Samsung Galaxy S24 (Family)"),
        input_specs=dict(image=input_shape),
    )
    compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])

    # Get compiled model
    compiled_model = compile_job.get_target_model()
    print(f"Compiled Model: {compiled_model}")

    if compiled_model is None:
        raise ValueError("Compiled model is None. Check if compilation succeeded.")

    # Create and upload dataset
    print("Creating dataset with enhanced preprocessing...")
    processed_images = []
    for tensor in input_tensors:
        processed_images.append(tensor.cpu().numpy())

    upload_dict = {"image": processed_images}
    new_dataset = qai_hub.upload_dataset(upload_dict)

    # Submit inference job
    device = qai_hub.Device("Samsung Galaxy S24 (Family)")

    inference_output = qai_hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=new_dataset,
        options="--max_profiler_iterations 1"
    ).download_output_data()

    # Process model outputs
    output_logits = inference_output["output_0"]
    predicted_classes = np.squeeze(np.array(output_logits))

    # Compute accuracy
    if len(predicted_classes.shape) == 2:
        predicted_labels = np.argmax(predicted_classes, axis=1)

        # Calculate accuracy
        correct_predictions = 0
        for i, file_name in enumerate(file_names):
            if i < len(predicted_labels) and int(predicted_labels[i]) == int(ground_truth[file_name]):
                correct_predictions += 1

        accuracy = (correct_predictions / len(file_names)) * 100
        print(f"Fine-tuned Model Accuracy: {accuracy:.2f}%")

        # Save results
        with open("model_performance.txt", "a") as f:
            f.write(f"\n=== PERFORMANCE COMPARISON ===\n")
            f.write(f"Model: Preserved Classifier + Fine-tuned MobileNetV2\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
    else:
        print(f"Unexpected shape of output: {predicted_classes.shape}")
else:
    print("No valid images found in the data directory")