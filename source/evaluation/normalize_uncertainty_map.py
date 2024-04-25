import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')

import numpy as np
import pandas
from utils import get_nii_data, get_nii_affine, save_image
from utils_uncertainty import normalize, min_max
from sklearn.metrics import recall_score, roc_auc_score, precision_score, auc, confusion_matrix
import pandas as pd

def main():

	n_subject = 41
	image_size = [256, 128, 256]

	# Get the name list of the subject
	list_subject_name_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
	data = pandas.read_excel(list_subject_name_dir, index=False)
	data_length = data.shape
	subject_names = np.array(data['subject_name'])
	affines = []

	# Get the uncertainty maps
	uncertainty_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp3/predictive_entropy/allVOI_cross_entropy_loss'
	uncertainty_maps = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	
	for n in range(n_subject):
		uncertainty_maps[n] =  get_nii_data(uncertainty_dir + '/' + subject_names[n] + '_predictiveEntropy.nii' )
		affines.append(get_nii_affine(uncertainty_dir + '/' + subject_names[n] + '_predictiveEntropy.nii' ))
		
	# Normalize the uncertainty maps by their maximum value inside the whole data set
	normalized_uncertainty_maps = normalize(uncertainty_maps)
	
	for n in range(n_subject):
		save_image(normalized_uncertainty_maps[n], affines[n], '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp3/normalized_uncertainty_map/' + subject_names[n] + '_normalized_uncertainty_map.nii')
		
	
if __name__ == '__main__':
    main()
	