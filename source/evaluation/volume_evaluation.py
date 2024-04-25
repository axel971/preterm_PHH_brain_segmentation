import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
import os
import numpy as np
from model.prediction_my import evaluate, test
from model.dataio import import_data_filename, write_nii
import metrics.metrics2 as metrics2
import metrics.metrics as metrics
import time
import pickle
import gc
import pandas as pd
from utils import get_nii_data, get_nii_affine, save_image, get_voxel_spacing



n_subject = 36
subject_list = np.arange(n_subject)
#image_size = [256, 128, 256]
labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels
    
print('toto and tata')
    
# Get path data
file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
data = pd.read_excel(file_dir, index=False)
data_length = data.shape
subject_names = np.array(data['subject_name'])
    
#delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_01/prediction/allVOI_cross_entropy_loss'
delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/prediction/allVOI_crossEntropy'
volume_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/volume/volume_manual.csv'
    
# Compute the Dice score for each subject
volume = []
    
for n in range(n_subject):
	#delineation = get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
	delineation = get_nii_data(delineation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii')
	voxel_spacing = get_voxel_spacing(delineation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii')
	#voxel_spacing = get_voxel_spacing(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
	volume.append(metrics.volume_multi_array(delineation, labels, voxel_spacing))
	
	
data = pd.DataFrame(volume)
data.to_csv(volume_dir)
    


    
