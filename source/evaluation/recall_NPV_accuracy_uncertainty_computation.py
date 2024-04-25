import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')

import numpy as np
import pandas
from utils import get_nii_data, get_nii_affine, save_image
from utils_uncertainty import normalize, min_max
from sklearn.metrics import recall_score, roc_auc_score, precision_score, auc, confusion_matrix, accuracy_score
import pandas as pd

def main():

	n_subject = 41
	image_size = [256, 128, 256]

	# Get the name list of the subject
	list_subject_name_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
	data = pandas.read_excel(list_subject_name_dir)
	data_length = data.shape
	subject_names = np.array(data['subject_name'])
	
	# Get the difference maps between the ground truth and the predicted segmentations
	difference_maps_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/difference_map/'
	difference_maps = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	affines = []
	for n in range(n_subject):
		difference_maps[n] =  get_nii_data(difference_maps_dir + '/' + subject_names[n] +  '_difference_map.nii')
		affines.append(get_nii_affine(difference_maps_dir  + '/' + subject_names[n] +  '_difference_map.nii'))
	

	# Get the uncertainty maps
	uncertainty_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/predictive_entropy/allVOI_cross_entropy_loss/'
	uncertainty_maps = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	
	for n in range(n_subject):
		uncertainty_maps[n] =  get_nii_data(uncertainty_dir + '/' + subject_names[n] + '_predictiveEntropy.nii' )
		
	# Normalize the uncertainty maps by their maximum value inside the whole data set
	normalized_uncertainty_maps = normalize(uncertainty_maps)
	
	# Applied a threshold on the normalized uncertainty maps
	recalls = np.zeros(n_subject, dtype = np.float)
	NPVs = np.zeros(n_subject, dtype = np.float)
	accuracies = np.zeros(n_subject, dtype = np.float)
	
	opt_threshold = 6.4
	
	for iSubject in range(n_subject):
		
		print(iSubject)
		thresh_uncertainty_maps = (normalized_uncertainty_maps[iSubject] >= (opt_threshold/100.00))
		recalls[iSubject] = recall_score(difference_maps[iSubject].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int), pos_label = 1)
		accuracies[iSubject] = accuracy_score(difference_maps[iSubject].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int))
		
		tn, fp, fn, tp = confusion_matrix(difference_maps[iSubject].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int)).ravel()
		NPVs[iSubject] = tn.astype(float)/(tn.astype(float) + fn.astype(float))
  		
		print('Recall: ' + str(recalls[iSubject]))
		print('NPV: ' + str(NPVs[iSubject]))
		print('Accuracy: ' + str(accuracies[iSubject]))
		
	
	data_recall = pd.DataFrame(recalls)
	data_recall.to_csv('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/recall_uncertainty/recall.csv')
	
	data_NPV = pd.DataFrame(NPVs)
	data_NPV.to_csv('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/NPV_uncertainty/NPV.csv')
	
	data_accuracy = pd.DataFrame(accuracies)
	data_accuracy.to_csv('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/accuracy_uncertainty/accuracy.csv')
    
	
if __name__ == '__main__':
    main()
	