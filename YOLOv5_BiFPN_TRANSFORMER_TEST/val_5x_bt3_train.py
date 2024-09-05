import os
import subprocess
import torch 
import yaml
import wget

# Download and get initial pretrained weights
file_url = 'https://drive.google.com/file/d/1Seu1BWeJbB7bcI7jj0uxubO9OWicleg6/view?usp=drive_link'
output_file = 'yolo_5x_bt_trained.pt'
wget.download(file_url, out=output_file)
print(f'Download complete. File saved as {output_file}')

# Create custom dataset file for final training
result = {
    "train": f"/gwhd_dataset/rs_train/images",
    "val": f"/gwhd_dataset/val/images",
    "test": f"/gwhd_dataset/test/images",
    "nc": 1,
    "names": ["wheat"],
}

with open("custom_dataset3.yaml", "w") as f:
    dump = yaml.dump(result, default_flow_style=False)
    f.write(dump)

# Create the directory
results_dir_path = 'results_5x_bt'

try:
	os.makedirs(results_dir_path, exist_ok=True) 
	print(f"Directory '{results_dir_path}' created successfully.")
except Exception as e:
	print(f"An error occurred: {e}")


# Specify weights from final training and path to custom dataset file
weights='yolo_5x_bt_trained.pt'
data_yaml = 'custom_dataset3.yaml'

val_output_dir = 'results_5x_bt/runs/val'
val_exp_name = 'exp_5x_bt3_train'

# Define the validation parameters
img_size = 800
batch_size = 8
conf_thres = 0.25
iou_thres = 0.45
task = 'train'

print("Validate the Model:")

# Command to execute the validation script
val_command = [
	'python', 'YOLOv5/val.py',
	'--weights', weights,
	'--data', data_yaml,
	'--img', str(img_size),
	'--batch', str(batch_size),
	'--conf-thres', str(conf_thres),
	'--iou-thres', str(iou_thres),
	'--save-txt',
	'--project', val_output_dir,
	'--name', val_exp_name,
	'--task', task,
	'--half'
]

# Open a file to write the output
with open('val_5x_bt_log3_train.txt', 'w') as val_log_file:
    result = subprocess.run(val_command, stdout=val_log_file, stderr=subprocess.STDOUT)

if result.returncode != 0:
    print(f"Validation command failed with return code {result.returncode}. See 'val_5x_bt_log3_train.txt' for details.")
else:
    print("Validation command executed successfully. See 'val_5x_bt_log3_train.txt' for details.")

print("Validating the Model Completed.")
print("See final metrics in val_5x_bt_log3_train.txt file")

