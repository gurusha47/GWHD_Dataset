import os
import pandas as pd
import random
import sys
import yaml
import fileinput
import shutil
import matplotlib
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.model_selection import KFold

# Path to dataset files
file_path= '/gwhd_2021/'

# Image folder path
image_folder = file_path + 'images'

print("Read Data into Dataframes")
# CSV files
train_csv = file_path + 'competition_train.csv'
val_csv = file_path + 'competition_val.csv'
test_csv = file_path + 'competition_test.csv'
metadata_csv = file_path + 'metadata_dataset.csv'

# Load CSV files
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)
metadata_df = pd.read_csv(metadata_csv, sep=';')

# Display the first few rows of Train DataFrame
print(f"Rows in Train Dataframe: {len(train_df)}")
print("Train DataFrame:")
print(train_df.head())

# Display the first few rows of Val DataFrame
print(f"Rows in Validation Dataframe: {len(val_df)}")
print("Validation DataFrame:")
print(val_df.head())

# Display the first few rows of Test DataFrame
print(f"Rows in Test Dataframe: {len(test_df)}")
print("Test DataFrame:")
print(test_df.head())

# Display first few rows of Metadata Dataframe
print("\nMetadata DataFrame:")
print(metadata_df.head())

# Create a eda directory to store visualizing images
viz_directory_path = '/gwhd_eda'

# Create the directory
try:
	os.makedirs(viz_directory_path, exist_ok=True)  # exist_ok=True avoids throwing an error if the directory already exists
	print(f"Directory '{viz_directory_path}' created successfully.")
except Exception as e:
	print(f"An error occurred: {e}")

# View a random image
print("Visualize and save a random image from Training Dataframe")
def visualize(image_row, image_folder, image_save_name):
	image_name = image_row['image_name']
	boxes = image_row['BoxesString']
	image_path = os.path.join(image_folder, image_name)
	image = cv2.imread(image_path)
	boxes = boxes.split(';')
	boxes = [list(map(int, box.split(' '))) for box in boxes]
	for (x,y,xx,yy) in boxes:
		cv2.rectangle(image,(int(x),int(y)),(int(xx),int(yy)),(255,0,0),5)
	plt.imshow(image)
	plt.axis('off')
	plt.title("Image in Training Dataframe")
	plt.savefig(f"{viz_directory_path}/{image_save_name}")
	plt.show()

i = random.randint(0, len(train_df))
visualize(train_df.iloc[i], image_folder, "Randam_Image_From_Training_Dataframe.png")

# Checking for missing values
print("Check for missing values in Dataframes")
print("Missing values in Dataframes:")
print(f"For Training Dataframe:\n{train_df.isnull().sum()}")
print(f"For Validation Dataframe:\n{val_df.isnull().sum()}")
print(f"For Testing Dataframe:\n{test_df.isnull().sum()}")

# Checking for unique images
print("Check for unique images in Dataframes")
print("Unique images in Dataframes")
print(f"No. of rows in Train Dataframe: {len(train_df)}")
print(f"No. of unique images in Train Dataframe: {len(train_df['image_name'].unique())}")
print(f"No. of rows in Validation Dataframe: {len(val_df)}")
print(f"No. of unique images in Validation Dataframe: {len(val_df['image_name'].unique())}")
print(f"No. of rows in Test Dataframe: {len(test_df)}")
print(f"No. of unique images in Test Dataframe: {len(test_df['image_name'].unique())}")

# Clean the training dataset of duplicate Images
print("Clean the Training Dataframe of duplicate Images")
names = list(set(train_df['image_name'].tolist()))
names_list = train_df['image_name'].tolist()
count_dict = dict(Counter(names_list))
for key in count_dict.keys():
	if count_dict[key] != 1:
		print(f"Image Name: {key}, Image Count: {count_dict[key]}")

target = 'd88963636d49127bda0597ef73f1703e92d6f111caefc44902d5932b8cd3fa94.png'
print(f"Image Name: {target}")
print("Index:")
for index, name in enumerate(names_list):
	if name == target:
		print(index)

target2 = '1961bcf453d5b2206c428c1c14fe55d1f26f3c655db0a2b6a83094476e8edb5b.png'
print(f"Image Name: {target2}")
print("Index:")
for index, name in enumerate(names_list):
	if name == target2:
		print(index)

print(f"Visualize and save Images with name: {target}")
visualize(train_df.iloc[1986], image_folder, "Train_Duplicate_Image_1_a.png")
visualize(train_df.iloc[2070], image_folder, "Train_Duplicate_Image_1_b.png")

print(f"Visualize and save Images with name: {target2}")
visualize(train_df.iloc[1999], image_folder, "Train_Duplicate_Image_2_a.png")
visualize(train_df.iloc[2079], image_folder, "Train_Duplicate_Image_2_b.png")

clean_train_df = train_df.drop(train_df.index[[1986, 1999]])
clean_train_df = clean_train_df.reset_index(drop=True)
clean_train_df.to_csv('competition_clean_train.csv', index=False)
shutil.move('competition_clean_train.csv', '/cs/home/psxss39/gwhd_2021')
print("Dupicate Images removed in clean_train_df and saved to file.")

# Clean the Testing dataset of duplicate Images
print("Clean the Testing Dataframe of duplicate Images")
test_names = list(set(test_df['image_name'].tolist()))
test_names_list = test_df['image_name'].tolist()
test_count_dict = dict(Counter(test_names_list))
for key in test_count_dict.keys():
	if test_count_dict[key] != 1:
		print(f"Image Name: {key}, Image Count: {test_count_dict[key]}")
	
target3 = 'da9846512ff19b8cd7278c8c973f75d36de8c4eb4e593b8285f6821aae1f4203.png'
print(f"Image Name: {target3}")
print("Index:")
for index, name in enumerate(test_names_list):
	if name == target3:
		print(index)

print(f"Visualize and save Images with name: {target3}")
visualize(train_df.iloc[911], image_folder, "Test_Duplicate_Image_1_a.png")
visualize(train_df.iloc[1038], image_folder, "Test_Duplicate_Image_1_b.png")

print("Image name is same but images are different. Hence dont need to clean the test_df.")

# Visualize Distribution of Domains in each dataset subset
print("Visualize and save Distribution of Domains in each dataset subset")
# Calculate value counts of the column
train_counts = clean_train_df['domain'].value_counts()
val_counts = val_df['domain'].value_counts()
test_counts = test_df['domain'].value_counts()

# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 7))
axs[0].pie(train_counts, labels=train_counts.index, autopct='%1.1f%%', startangle=140)
axs[0].set_title('Image Domains in Train Dataset')
axs[0].axis('equal')
axs[1].pie(val_counts, labels=val_counts.index, autopct='%1.1f%%', startangle=140)
axs[1].set_title('Image Domains in Val Dataset')
axs[1].axis('equal')
axs[2].pie(test_counts, labels=test_counts.index, autopct='%1.1f%%', startangle=140)
axs[2].set_title('Image Domains in Test Dataset')
axs[2].axis('equal')
plt.savefig(f"{viz_directory_path}/Domain_Distributions_in_Dataframes.png")
plt.show()
