import os
import subprocess
import torch 

weights='yolo_5x_bt_trained.pt'
data_source = '/cs/home/psxss39/gwhd_dataset/test/images'
detect_output_dir = '/cs/home/psxss39/gwhd_results/runs/detect'
detect_exp_name = 'exp_5x_bt3'

# Define the validation parameters
img_size = 800
batch_size = 8
conf_thres = 0.25
iou_thres = 0.45

print("Testing the Model:")

# Command to execute the validation script
detect_command = [
	'python', 'YOLOv5/detect.py',
	'--weights', weights,
	'--source', data_source,
	'--img', str(img_size),
	'--conf-thres', str(conf_thres),
	'--iou-thres', str(iou_thres),
	'--nosave',
	'--save-txt',
	'--project', detect_output_dir,
	'--name', detect_exp_name,
]

# Open a file to write the output
with open('detect_5x_bt_log3.txt', 'w') as detect_log_file:
    # Run the command and redirect stdout and stderr to the log file
    result = subprocess.run(detect_command, stdout=detect_log_file, stderr=subprocess.STDOUT)

# Check the return code to see if the command was successful
if result.returncode != 0:
    print(f"Testing command failed with return code {result.returncode}. See 'detect_5x_bt_log3.txt' for details.")
else:
    print("Testing command executed successfully. See 'detect_5x_bt_log3.txt' for details.")

print("Testing the Model Completed.")
print("See final metrics in detect_5x_bt_log3.txt file")

