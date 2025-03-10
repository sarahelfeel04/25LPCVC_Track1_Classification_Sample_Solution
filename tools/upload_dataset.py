import os
import numpy as np
from PIL import Image
import qai_hub

def process_image(image_path, input_shape):
    """Load and process an image from a local path to the required input shape."""
    image = Image.open(image_path).convert('RGB').resize((input_shape[2], input_shape[3]))
    input_array = np.expand_dims(
        np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
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

dataset = qai_hub.upload_dataset({"image": input_data})

if dataset:
    dataset_id = dataset.id  # Extract the ID as a string
    print(f"Dataset uploaded successfully! Dataset ID: {dataset_id}")
    
    # Write dataset ID to a file
    with open("dataset_id.txt", "w") as f:
        f.write(dataset_id)
else:
    print("Dataset upload failed!")