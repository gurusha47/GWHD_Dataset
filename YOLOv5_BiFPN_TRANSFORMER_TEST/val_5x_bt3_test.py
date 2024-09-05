import os
import subprocess
import torch 
import yaml
import wget

# Specify weights from final training and path to custom dataset file
weights='yolo_5x_bt_trained.pt'
data_yaml = 'custom_dataset3.yaml'

val_output_dir = 'results_5x_bt/runs/val'
val_exp_name = 'exp_5x_bt3_test'

# Define the validation parameters
img_size = 800
batch_size = 8
conf_thres = 0.25
iou_thres = 0.45
task = 'test'

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
with open('val_5x_bt_log3_test.txt', 'w') as val_log_file:
    result = subprocess.run(val_command, stdout=val_log_file, stderr=subprocess.STDOUT)

if result.returncode != 0:
    print(f"Validation command failed with return code {result.returncode}. See 'val_5x_bt_log3_test.txt' for details.")
else:
    print("Validation command executed successfully. See 'val_5x_bt_log3_test.txt' for details.")

print("Validating the Model Completed.")
print("See final metrics in val_5x_bt_log3_test.txt file")

