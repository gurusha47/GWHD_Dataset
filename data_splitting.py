import os
import pandas as pd
import random
import sys
import yaml
import fileinput
import shutil
import cv2
from collections import Counter

# Path to dataset files
file_path= '/gwhd_2021/'

# Image folder path
image_folder = file_path + 'images'

print("Read Data into Dataframes")
# CSV files
train_csv = file_path + 'competition_clean_train.csv'
val_csv = file_path + 'competition_val.csv'
test_csv = file_path + 'competition_test.csv'
metadata_csv = file_path + 'metadata_dataset.csv'

# Load CSV files
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)
metadata_df = pd.read_csv(metadata_csv, sep=';')

print("Rearrange Images and Labels in directories")

train_image_dir = '/gwhd_dataset/train/images'
val_image_dir = '/gwhd_dataset/val/images'
test_image_dir = '/gwhd_dataset/test/images'
train_label_dir = '/gwhd_dataset/train/labels'
val_label_dir = '/gwhd_dataset/val/labels'
test_label_dir = '/gwhd_dataset/test/labels'

# Ensure the destination directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

def move_images(df, dest_image_dir, dest_label_dir):
	for idx, row in df.iterrows():
		image_name = row['image_name']
		boxesstring = row['BoxesString']
		label_name = image_name.replace('.png', '.txt')
		
		source_path = os.path.join(image_folder, image_name)
		
		dest_image_path = os.path.join(dest_image_dir, image_name)
		shutil.copy(source_path, dest_image_path)
		
		dest_label_path = os.path.join(dest_label_dir, label_name)
		with open(dest_label_path, 'w') as label_file:
			label_file.write(boxesstring)


# Move images to respective directories
move_images(train_df, train_image_dir, train_label_dir)
move_images(val_df, val_image_dir, val_label_dir)
move_images(test_df, test_image_dir, test_label_dir)

# Update the Labels as per YOLO format
print("Update the Labels as per YOLO format")
def convert_box_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
	# Convert to YOLO format
	center_x = (x_min + x_max) / 2.0 / img_width
	center_y = (y_min + y_max) / 2.0 / img_height
	width = (x_max - x_min) / img_width
	height = (y_max - y_min) / img_height
	return center_x, center_y, width, height

def process_label_file(label_path, img_width, img_height):
	yolo_labels = []
	
	with open(label_path, 'r') as file:
		lines = file.readlines()
	
	for line in lines:
		# Example format: "20 976 96 1024;308 942 488 1024;..."
		boxes = line.strip().split(';')
		for box in boxes:
			parts = box.split()
			if len(parts) == 4:
				x_min = float(parts[0])
				y_min = float(parts[1])
				x_max = float(parts[2])
				y_max = float(parts[3])
				center_x, center_y, width, height = convert_box_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height)
				yolo_labels.append(f"{0} {center_x} {center_y} {width} {height}")
	return '\n'.join(yolo_labels)
	
def update_labels(dir):
	for root, _, files in os.walk(dir):
		for file in files:
			if file.endswith('.txt'):
				label_path = os.path.join(root, file)
				# Convert and save new label format
				yolo_labels = process_label_file(label_path, 1024, 1024)
				with open(label_path, 'w') as f:
					f.write(yolo_labels)

update_labels(dir=train_label_dir)
update_labels(dir=val_label_dir)
update_labels(dir=test_label_dir)

print("All labels updated successfully in their respective directories")

dest_images_folder = '/gwhd_dataset/rs_train/images'
dest_labels_folder = '/gwhd_dataset/rs_train/labels'
os.makedirs(dest_images_folder, exist_ok=True)
os.makedirs(dest_labels_folder, exist_ok=True)

image_src = '/gwhd_dataset/train/images'
label_src = '/gwhd_dataset/train/labels'

all_images = [f for f in os.listdir(image_src) if f.endswith('.png')]
random.shuffle(all_images)

for i, image in enumerate(all_images):
    label = image.replace('.png', '.txt')

    # Move to the destination
    shutil.move(os.path.join(image_src, image), os.path.join(dest_images_folder, image))
    shutil.move(os.path.join(label_src, label), os.path.join(dest_labels_folder, label))

print("Random shuffling of trining datset complete")

# Read CSV file
clean_train_csv = '/gwhd_2021/competition_clean_train.csv'
train_df = pd.read_csv(clean_train_csv)

# Create domain mapping
domain_mapping = {domain: i for i, domain in enumerate(train_df['domain'].unique())}
train_df['domain_id'] = train_df['domain'].map(domain_mapping)

# Get domain count
count_dict = dict(train_df["domain_id"].value_counts())

def get_split(ratio=3, bound=10):
    total_samples = len(train_df)
    target_train = total_samples * (ratio / (ratio + 1))
    target_val = total_samples * (1 / (ratio + 1))
    
    while True:
        domain_list = [i for i in range(len(domain_mapping))]
        random.shuffle(domain_list)
        
        domains_0 = domain_list[:int(len(domain_list) * (ratio / (ratio + 1)))]
        domains_1 = domain_list[int(len(domain_list) * (ratio / (ratio + 1))):]
        
        def get_count(l):
            return sum(count_dict[element] for element in l)
        
        count_train = get_count(domains_0)
        count_val = get_count(domains_1)
        
        if (-bound < target_train - count_train < bound and -bound < target_val - count_val < bound):
            print("Train domains:", domains_0, "Validation domains:", domains_1)
            print("Train count difference:", target_train - count_train, "Validation count difference:", target_val - count_val)
            return domains_0, domains_1

# Get split values with 3:1 ratio
get_split_value = get_split(ratio=3, bound=7)

domains_0, domains_1 = get_split_value

# Generate train and validation indexes
train_indexes = [idx for domain in domains_0 for idx in train_df.index[train_df["domain_id"] == domain]]
val_indexes = [idx for domain in domains_1 for idx in train_df.index[train_df["domain_id"] == domain]]

# Print lengths
print("Number of training samples:", len(train_indexes))
print("Number of validation samples:", len(val_indexes))
print("Total samples:", len(train_indexes + val_indexes))

dest_image_train_dir = '/gwhd_dataset/ds_train/images/train'
dest_label_train_dir = '/gwhd_dataset/ds_train/labels/train'

dest_image_val_dir = '/gwhd_dataset/ds_train/images/val'
dest_label_val_dir = '/gwhd_dataset/ds_train/labels/val'

os.makedirs(dest_image_train_dir, exist_ok=True)
os.makedirs(dest_label_train_dir, exist_ok=True)

os.makedirs(dest_image_val_dir, exist_ok=True)
os.makedirs(dest_label_val_dir, exist_ok=True)

for idx in train_indexes:
    image_name = train_df.iloc[idx]['image_name']
    boxesstring = train_df.iloc[idx]['BoxesString']
    label_name = image_name.replace('.png', '.txt')
    
    source_path = os.path.join('/gwhd_dataset/train/images', image_name)
    
    dest_image_path = os.path.join(dest_image_train_dir, image_name)
    shutil.copy(source_path, dest_image_path)
    
    dest_label_path = os.path.join(dest_label_train_dir, label_name)
    with open(dest_label_path, 'w') as label_file:
        label_file.write(boxesstring)

for idx in val_indexes:
    image_name = train_df.iloc[idx]['image_name']
    boxesstring = train_df.iloc[idx]['BoxesString']
    label_name = image_name.replace('.png', '.txt')
    
    source_path = os.path.join('/gwhd_dataset/train/images', image_name)
    
    dest_image_path = os.path.join(dest_image_val_dir, image_name)
    shutil.copy(source_path, dest_image_path)
    
    dest_label_path = os.path.join(dest_label_val_dir, label_name)
    with open(dest_label_path, 'w') as label_file:
        label_file.write(boxesstring)

print("Random domain split for training int train and val split")

# Update the Labels as per YOLO format
print("Update the Labels as per YOLO format")
def convert_box_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
	# Convert to YOLO format
	center_x = (x_min + x_max) / 2.0 / img_width
	center_y = (y_min + y_max) / 2.0 / img_height
	width = (x_max - x_min) / img_width
	height = (y_max - y_min) / img_height
	return center_x, center_y, width, height

def process_label_file(label_path, img_width, img_height):
	yolo_labels = []
	
	with open(label_path, 'r') as file:
		lines = file.readlines()
	
	for line in lines:
		# Example format: "20 976 96 1024;308 942 488 1024;..."
		boxes = line.strip().split(';')
		for box in boxes:
			parts = box.split()
			if len(parts) == 4:
				x_min = float(parts[0])
				y_min = float(parts[1])
				x_max = float(parts[2])
				y_max = float(parts[3])
				center_x, center_y, width, height = convert_box_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height)
				yolo_labels.append(f"{0} {center_x} {center_y} {width} {height}")
	return '\n'.join(yolo_labels)
	
def update_labels(dir):
	for root, _, files in os.walk(dir):
		for file in files:
			if file.endswith('.txt'):
				label_path = os.path.join(root, file)
				# Convert and save new label format
				yolo_labels = process_label_file(label_path, 1024, 1024)
				with open(label_path, 'w') as f:
					f.write(yolo_labels)

update_labels(dir=dest_label_train_dir)
update_labels(dir=dest_label_val_dir)

print("All labels updated successfully in train and val split directories")