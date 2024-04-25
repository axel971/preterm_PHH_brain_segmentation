import sys
sys.path.append('..')

import numpy as np
import os
import pandas
from scipy.ndimage import zoom
#from skimage.transform import resize
from utils import get_nii_data
import cv2


# Get the patient name (ID) from the xlsx file
path_subject_name = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.xlsx"
data = pandas.read_excel(path_subject_name, index=False)
data_length = data.shape

subject_names = np.array(data['subject_name'])

# Instantiate the input and output path
delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/raw'
delineation_preproc_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/relabeled'
img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/raw'
img_preproc_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing'

# Create a directory for the preprocessing delineation 
try:
	#Create the directory
	os.mkdir(delineation_preproc_dir)
except FileExistsError:
	print('Directory ', delineation_preproc_dir, ' already exists')
        
# relabeling the delineations
print('Relabel the delineations')
for subject_name in subject_names:
	cmd = './relabeling/build/relabeling ' + delineation_dir + '/' + subject_name + '_delineations.nii.gz ' +  delineation_preproc_dir + '/' + subject_name + '_delineations.nii.gz'
	os.system(cmd)
	print(cmd)

# Create a directory for the preprocessed images
try:
	#Create the directory
	os.mkdir(img_preproc_dir)
except FileExistsError:
	print('Directory ', img_preproc_dir, ' already exists')
	
# Bias field correction
print('Bias field correction')
for subject_name in subject_names:
	cmd = './bias_field_correction/build/bias_field_correction ' + img_dir + '/' + subject_name + '.nii.gz '  +  delineation_preproc_dir + '/' + subject_name + '_delineations.nii.gz ' +  img_preproc_dir + '/' + subject_name + '_preproc.nii.gz'
	os.system(cmd)
	print(cmd)
	
# Resampling of the images
print('Resampling')
for subject_name in subject_names:
	cmd = './resampling/build/resampling ' + img_preproc_dir + '/' + subject_name + '_preproc.nii.gz'  +  img_preproc_dir + '/' + subject_name + '_preproc.nii.gz'
	os.system(cmd)
	print(cmd)
	
# Compute gradients of the images
# print('Compute the gradient of the images')
# for patient_name in patient_names:
# 	cmd = './gradientImage/build/gradientImage '  + img_preproc_dir + '/' + patient_name + '_preproc.nii.gz' + ' ' + img_preproc_dir + '/' + patient_name + '_preproc.nii.gz'
# 	os.system(cmd)
# 	print(cmd)

