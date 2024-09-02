import os
import subprocess
import torch 
import yaml

# Specify weights from initial training and path to custom dataset file
weights=os.path.join('results_5x_bifpn/runs/train', 'exp_5x_bifpn1', 'weights', 'best.pt')
data_yaml_path = 'custom_dataset1.yaml'

# Specify validation output directory
val_output_dir = 'results_5x_bifpn/runs/val'
val_exp_name = 'exp_5x_bifpn1'

# Define the validation parameters
img_size = 800
batch_size = 8
conf_thres = 0.25
iou_thres = 0.45
task1 = 'train'
task2 = 'val'

print("Validate the Model:")

# Command to execute the validation script
val_command = [
	'python', 'YOLOv5/val.py',
	'--weights', weights,
	'--data', data_yaml_path,
	'--img', str(img_size),
	'--batch', str(batch_size),
	'--conf-thres', str(conf_thres),
	'--iou-thres', str(iou_thres),
	'--save-txt',
	'--project', val_output_dir,
	'--name', val_exp_name,
	'--task', task1,
	#'--task', task2,
	'--half'
]

# Open a file to write the output
with open('val_5x_bifpn_log1.txt', 'w') as val_log_file:
    result = subprocess.run(val_command, stdout=val_log_file, stderr=subprocess.STDOUT)

if result.returncode != 0:
    print(f"Validation command failed with return code {result.returncode}. See 'val_5x_bifpn_log1.txt' for details.")
else:
    print("Validation command executed successfully. See 'val_5x_bifpn_log1.txt' for details.")

print("Validating the Model Completed.")
print("See final metrics in val_5x_bifpn_log1.txt file")

