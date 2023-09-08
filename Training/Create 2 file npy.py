import os
import cv2
import numpy as np
import time

#The function is created to calculate the total number of image files in the folder "Raw Dataset"
#The function reads all subfolders in the "Raw Dataset" folder and creates a list of image files in each subfolder.
def total_files():
    total = 0
    for i in MAIN_DIR:
        train_test_base = os.path.join(BASE, DATASET_FOLDER_NAME, i)
        train_test_dir = os.listdir(train_test_base)
        for j in train_test_dir:
            labeled_dir_path = os.path.join(train_test_base, j)
            all_img = os.listdir(labeled_dir_path)
            total += len(all_img)
    print(f'Total files are {total}')

#Image data from the "with_mask" and "without_mask" subfolders will be moved to the list,
#and the corresponding labels (1 for "with_mask" and 0 for "without_mask")
target = []
data = []
data_map = {
    'with_mask': 1,
    'without_mask': 0
}
skipped = 0


BASE = "C:/Users/kimng/Downloads/Face-mask"
DATASET_FOLDER_NAME = "Raw Dataset"
img_shape = 50


TRAINING_FOLDER = os.path.join(os.getcwd(), "Training")
if not os.path.exists(TRAINING_FOLDER):
    os.makedirs(TRAINING_FOLDER)


MAIN_DIR = os.listdir(os.path.join(BASE, DATASET_FOLDER_NAME))

#this function reads image files, resizes images, collects data and labels respectively into two numpy
total_files()
for i in MAIN_DIR:
    train_test_base = os.path.join(BASE, DATASET_FOLDER_NAME, i)
    train_test_dir = os.listdir(train_test_base)
    for j in train_test_dir:
        labeled_dir_path = os.path.join(train_test_base, j)
        all_img = os.listdir(labeled_dir_path)
        print(f'\nExecuting - {i}/{j}')
        for k in all_img:
            image_path = os.path.join(labeled_dir_path, k)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            try:
                #Image data from the "with_mask" and "without_mask" files will be read and resized to 50x50 pixels.
                image = cv2.resize(image, (img_shape, img_shape))
            except Exception as E:
                skipped += 1
                print(E)
                continue
            #Then the picture and label (1 for "with_mask" and 0 for "without_mask") respectively will be added to the two arrays
            data.append(image)
            target.append(data_map[j])

print(f'\n{skipped} files skipped.')
#Finally, the array and will be converted into two numpy arrays for use during model training
data = np.array(data)
target = np.array(target)


#This snippet saves the processed image data and their corresponding labels into two separate files.
data_file_path = os.path.join(TRAINING_FOLDER, 'data.npy')
target_file_path = os.path.join(TRAINING_FOLDER, 'target.npy')

#The file will contain image data (variable) and the file will contain the corresponding label
np.save(data_file_path, data)
np.save(target_file_path, target)

#After the data has been saved, the program prints out a message to notify that the data has been saved successfully and displays the path to two files:
print('\nData file saved:', data_file_path)
print('Target file saved:', target_file_path)
print('\nFinished')
time.sleep(6000)
