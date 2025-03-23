import os
import numpy as np
import cv2
from PIL import Image
import qai_hub

def process_image(image_path, input_shape):
    """Load and process an image by averaging normal and equalized versions."""
    # Load and resize the image
    image = Image.open(image_path).convert('RGB').resize((input_shape[2], input_shape[3]))
    image_np = np.array(image, dtype=np.uint8)  # Convert to NumPy array

    # 1. Normal Image (Scaled)
    normal_image = image_np.astype(np.float32) / 255.0  # Normalize to [0,1]

    # 2. Histogram Equalization (CLAHE)
    image_eq = np.zeros_like(image_np)  # Placeholder for equalized image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):  # Apply CLAHE to each channel separately
        image_eq[:, :, i] = clahe.apply(image_np[:, :, i])
    
    # Normalize equalized image
    image_eq = image_eq.astype(np.float32) / 255.0

    # 3. Average of Normal and Equalized Image
    averaged_image = (normal_image + image_eq) / 2.0  # Element-wise average

    # 4. Reshape to (1, 3, 224, 224)
    input_array = np.expand_dims(
        np.transpose(averaged_image, (2, 0, 1)), axis=0
    )
    return input_array

# Set the local path where you manually downloaded the dataset
local_folder_path = "dataset/images"  # Change this to your actual path

# Get a list of all image files and sort them by name (to maintain order)
sample_image_paths = sorted([os.path.join(local_folder_path, filename) for filename in os.listdir(local_folder_path)
                             if filename.endswith(('.jpg', '.png', '.jpeg'))])

# Specify the input shape
input_shape = (len(sample_image_paths), 3, 224, 224)

# Process the images
input_data = [process_image(path, input_shape) for path in sample_image_paths]

print(f"Total images: {len(input_data)}")
print(f"First image shape: {input_data[0].shape}")

# Upload the dataset
dataset = qai_hub.upload_dataset({"image": input_data})

if dataset:
    dataset_id = dataset.id if hasattr(dataset, "id") else dataset.dataset_id
    print(f"Dataset uploaded successfully! Dataset ID: {dataset_id}")

    # Write dataset ID to a file
    with open("dataset_id.txt", "w") as f:
        f.write(str(dataset_id))
else:
    print("Dataset upload failed!")
