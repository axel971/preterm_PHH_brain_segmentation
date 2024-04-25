
import sys
sys.path.append('..')


from utils import get_nii_data
import numpy as np
import pandas


# Get the image quality label and patient name (ID) from the xlsx file
file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/images/subject_case_name.xlsx"
data = pandas.read_excel(file_dir, index=False)
data_length = data.shape

subject_names = np.array(data['subject_name'])

 
        
# Load the training images from the patient name 
img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case'
images = []
for subject_name in subject_names:
		images.append(get_nii_data(img_dir + '/' + subject_name + '_delineations.nii.gz' ))          
		print(get_nii_data(img_dir + '/' + subject_name + '_delineations.nii.gz' ).shape)



