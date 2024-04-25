import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')

import os
import numpy as np
from model.prediction_my import evaluate, test
from model.dataio import import_data_filename, write_nii
import metrics.metrics2 as metrics2
import time
import pickle
import gc
import pandas as pd
from utils import get_nii_data, get_nii_affine, save_image, get_voxel_spacing


n_subject = 41
subject_list = np.arange(n_subject)
labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels
    
print('Process: compute assd')
    
# Get path data
file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
data = pd.read_excel(file_dir, index=False)
data_length = data.shape
subject_names = np.array(data['subject_name'])
    
pred_delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/prediction/allVOI_crossEntropy'
delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
assd_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/assd/assd.csv'


# pred_delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/prediction/allVOI_crossEntropy'
# delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
# assd_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/assd/assd.csv'
   

# Compute the Dice score for each subject
assd = []
    
for n in range(n_subject):
	print(n)
	delineation1 = get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
	delineation2 = get_nii_data(pred_delineation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii')
	voxel_spacing = get_voxel_spacing(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
	assd.append(metrics2.assd_multi_array(delineation1, delineation2, labels, voxel_spacing))
	
	
data = pd.DataFrame(assd)
# data['Dice'] = dice
data.to_csv(assd_dir)
    


    
