import numpy as np
import os
import pandas
import sys
sys.path.append('..')
from utils import get_nii_data


# Get the image quality label and patient name from the xlsx file
file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.xlsx"
data = pandas.read_excel(file_dir, index=False)
data_length = data.shape

subject_names = np.array(data['subject_name'])
 
        
# Load the training images from the patient names 
img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/raw'
images = []
shapes = []
for subject_name in subject_names:
 	 images.append(get_nii_data(img_dir + '/' + subject_name + '.nii.gz' )) 
 	 shapes.append(get_nii_data(img_dir + '/' + subject_name + '.nii.gz').shape[2])       
 	 
print(np.amax(shapes))
print(np.amin(shapes))
#print(shapes)