import os
import shutil
from sklearn.model_selection import train_test_split

# Define the source and destination directories
src_dir = '/Users/reetvikchatterjee/Desktop/Dataset'  # Source directory containing the images
train_dir = '/Users/reetvikchatterjee/Desktop/SplitData/train'  # Destination directory for train data
valid_dir = '/Users/reetvikchatterjee/Desktop/SplitData/valid'  # Destination directory for validation data

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Loop over each label folder in the source directory
for label in os.listdir(src_dir):
    label_path = os.path.join(src_dir, label)
    
    if os.path.isdir(label_path):
        # Create label folders in train and valid directories
        train_label_dir = os.path.join(train_dir, label)
        valid_label_dir = os.path.join(valid_dir, label)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(valid_label_dir, exist_ok=True)
        
        # Get all image files in the label folder
        image_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
        
        # Split the data into train and validation sets (80% train, 20% valid)
        train_files, valid_files = train_test_split(image_files, test_size=0.2, random_state=42)
        
        # Move images to the respective folders
        for img in train_files:
            shutil.move(os.path.join(label_path, img), os.path.join(train_label_dir, img))
        
        for img in valid_files:
            shutil.move(os.path.join(label_path, img), os.path.join(valid_label_dir, img))

print("Dataset split into train and valid folders successfully!")
