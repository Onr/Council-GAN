import shutil
import os
import numpy as np
from tqdm import tqdm
input_folder = './MICCAI_BraTS2020_TrainingData'
output_folder_train = './MICCAI_BraTS2020_TrainingData_my_train'
output_folder_test = './MICCAI_BraTS2020_TrainingData_my_test'
ratio_of_test_images = 1/10


list_of_brain_folders = os.listdir(input_folder)
list_of_brain_folders = np.random.permutation(list_of_brain_folders)
list_of_brain_folders = [brain_folder for brain_folder in list_of_brain_folders if
                         os.path.isdir(os.path.join(input_folder, brain_folder))]

num_of_brains = len(list_of_brain_folders)
num_of_brains_test = int(num_of_brains * ratio_of_test_images)
list_of_brain_folders_test = list_of_brain_folders[:num_of_brains_test]
list_of_brain_folders_train = list_of_brain_folders[num_of_brains_test:]

print(f'totla brains: {num_of_brains}, train brains{len(list_of_brain_folders_train)}, test brains:{len(list_of_brain_folders_test)}')

if not os.path.exists(output_folder_train):
    os.mkdir(output_folder_train)
if not os.path.exists(output_folder_test):
    os.mkdir(output_folder_test)

for brain_folder_train in tqdm(list_of_brain_folders_train):
    sorce = os.path.join(input_folder, brain_folder_train)
    dest = os.path.join(output_folder_train, brain_folder_train)
    shutil.copytree(sorce, dest)

for brain_folder_test in tqdm(list_of_brain_folders_test):
    sorce = os.path.join(input_folder, brain_folder_test)
    dest = os.path.join(output_folder_test, brain_folder_test)
    shutil.copytree(sorce, dest)






