import os
import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')

import numpy as np
from model.image_process import crop_edge_pair, load_image_correct_oritation
from model.dataio import import_data_filename, write_nii
#from model.one_hot_label import redefine_label
# import metrics.metrics as
import time
import pickle
import gc
import uncertainty

import pandas

from utils import get_nii_data, get_nii_affine, save_image, get_voxel_spacing
from imgaug import augmenters as iaa

def main():
	
	n_subject = 41
	file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
	data = pandas.read_excel(file_dir)
	data_length = data.shape
	subject_names = np.array(data['subject_name'])
    
	MCsamples_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/MCsamples/allVOI_cross_entropy_loss'
	image_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing' 
	predictive_entropy_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/predictive_entropy/allVOI_cross_entropy_loss/'
	
	image_size = [256, 128, 256]
	patch_size = [64, 64, 64]
	labels = [0, 1, 2, 3, 4, 5, 6]
	
	for iSubject in range(n_subject):
		print("Processing subject: " + str(iSubject))
		affine = get_nii_affine(image_dir + '/' + subject_names[iSubject] + '_preproc.nii.gz')
		voxel_spacing = get_voxel_spacing(image_dir + '/' + subject_names[iSubject] + '_preproc.nii.gz')

		MCsamples = np.load(MCsamples_dir + '/' + subject_names[iSubject] + '_MCsamples.npy')
		predictive_entropy = uncertainty.predictive_entropy(MCsamples, image_size, labels)
		save_image(predictive_entropy, affine, predictive_entropy_dir + '/' + subject_names[iSubject] + '_predictiveEntropy.nii') 

		
if __name__ == '__main__':
    main()
