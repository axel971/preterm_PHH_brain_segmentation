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
#image_size = [256, 128, 256]
labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels
    
print('Process: compute hd 95th')
    
# Get path data
file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
data = pd.read_excel(file_dir)
data_length = data.shape
subject_names = np.array(data['subject_name'])
    
pred_delineation_dir =  '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/prediction/allVOI_crossEntropy'
delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
hd95_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/hd95/hd95.csv'

# pred_delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/prediction/allVOI_crossEntropy'
# delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
# hd95_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/hd95/hd95.csv'
    
# Compute the Dice score for each subject
hd95 = []
    
for n in range(n_subject):
	delineation1 = get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
	delineation2 = get_nii_data(pred_delineation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii')
	voxel_spacing = get_voxel_spacing(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
	hd95.append(metrics2.hd95_multi_array(delineation1, delineation2, labels, voxel_spacing))
	
	
data = pd.DataFrame(hd95)
data.to_csv(hd95_dir)
    


    
