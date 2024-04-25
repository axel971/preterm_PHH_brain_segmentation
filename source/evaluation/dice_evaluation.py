import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')

from utils import get_nii_data, get_nii_affine, save_image
import os
import numpy as np
from model.dataio import import_data_filename, write_nii
import metrics.metrics as metrics
import time
import pickle
import gc
import pandas as pd




n_subject = 41
subject_list = np.arange(n_subject)
labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels
    
print('Process: Dice computation')
    
# Get path data
file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
data = pd.read_excel(file_dir)
data_length = data.shape
subject_names = np.array(data['subject_name'])

# pred_delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/prediction/allVOI_crossEntropy'
pred_delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/prediction/allVOI_crossEntropy'
delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
dice_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/dice/dice.csv'
# dice_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/dice/dice.csv'
    
# Compute the Dice score for each subject
dice = []
    
for n in range(n_subject):
    dice.append(metrics.dice_multi_array(get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz'), get_nii_data(pred_delineation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii'), labels))
	
	
data = pd.DataFrame(dice)
# data['Dice'] = dice
data.to_csv(dice_dir)
    


    
