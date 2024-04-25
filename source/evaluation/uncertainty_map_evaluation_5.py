import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')

import numpy as np
import pandas
from utils import get_nii_data, get_nii_affine, save_image
from utils_uncertainty import normalize, min_max
from sklearn.metrics import recall_score, roc_auc_score, precision_score, auc, confusion_matrix
import pandas as pd

def main():

	eps = 1e-15 # Epsilon value to avoid division by zeros
	n_subject = 41
	image_size = [256, 128, 256]

	# Get the name list of the subject
	list_subject_name_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
	data = pandas.read_excel(list_subject_name_dir)
	data_length = data.shape
	subject_names = np.array(data['subject_name'])
	
	# Get the ground truth segmentation
	gt_segmentation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
	gt_segmentation = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	affines = []
	for n in range(n_subject):
		gt_segmentation[n] =  get_nii_data(gt_segmentation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
		affines.append(get_nii_affine(gt_segmentation_dir + '/' + subject_names[n] + '_delineations.nii.gz'))
# 		gt_segmentation[gt_segmentation == 7] = 0 # Hack: the ground segmentations have one volume in add compared to the predicted segmentation	
		
	# Get the predicted segmentation
	predicted_segmentation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/prediction/allVOI_cross_entropy_loss'
	predicted_segmentation = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	for n in range(n_subject):
		predicted_segmentation[n] =  get_nii_data(predicted_segmentation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii')
		
		
	# Compute the difference maps between the ground truth and the predicted segmentations
	difference_maps = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
	for iSubject in range(0, n_subject):
		difference_maps[iSubject] = (gt_segmentation[iSubject] != predicted_segmentation[iSubject])
		save_image(difference_maps[iSubject], affines[iSubject], '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/difference_map/' + subject_names[iSubject] + '_difference_map.nii')
    	
	# Get the uncertainty maps
	uncertainty_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/predictive_entropy/allVOI_cross_entropy_loss/'
	uncertainty_maps = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
	
	for n in range(n_subject):
		uncertainty_maps[n] =  get_nii_data(uncertainty_dir + '/' + subject_names[n] + '_predictiveEntropy.nii' )
		
	# Normalize the uncertainty maps by their maximum value inside the whole data set
	normalized_uncertainty_maps = normalize(uncertainty_maps)
	
	# Applied a threshold on the normalized uncertainty maps
	AUCs = np.zeros(n_subject, dtype = np.float)
	mean_recall = np.zeros(100, dtype = np.float)
	mean_FPR = np.zeros(100, dtype = np.float)
	opt_threshold = np.zeros(n_subject, dtype = np.float)
	
	for iSubject in range(n_subject):
		
		print(iSubject)
		
		recall = np.zeros(100, dtype= np.float32)
		FPR = np.zeros(100, dtype= np.float32)
		
		for iThreshold in range(0, 100):
  			thresh_uncertainty_maps = (normalized_uncertainty_maps[iSubject] >= (iThreshold/100.00))
  			recall[iThreshold] = recall_score(difference_maps[iSubject].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int), pos_label = 1)
  			# precision[iThreshold] = precision_score(difference_maps[0].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int), pos_label = 1)
  			tn, fp, fn, tp = confusion_matrix(difference_maps[iSubject].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int)).ravel()
  			FPR[iThreshold] = fp.astype(float)/(fp.astype(float) + tn.astype(float) + eps)
  		
		AUCs[iSubject] = auc(FPR, recall)
		print('AUC: ' + str(AUCs[iSubject]))
		
		gmean = np.sqrt(recall * (1 - FPR))
		opt_threshold[iSubject] = np.argmax(gmean)
		print('Optimal threshold: ' + str(opt_threshold[iSubject]))
		
		mean_recall += recall
		mean_FPR += FPR
		
	mean_recall /= n_subject
	mean_FPR /= n_subject

	data_AUC = pd.DataFrame(AUCs)
	data_AUC.to_csv('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/AUC_uncertainty/uncertainty_AUC.csv')
	
	data_mean_recall = pd.DataFrame(mean_recall)
	data_mean_recall.to_csv('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/mean_recall_uncertainty/mean_recall.csv')
	
	data_mean_FPR = pd.DataFrame(mean_FPR)
	data_mean_FPR.to_csv('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/mean_FPR_uncertainty/mean_FPR.csv')
    
    	
	data_optimal_threshold = pd.DataFrame(opt_threshold)
	data_optimal_threshold.to_csv('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_05/opt_threshold_uncertainty/opt_threshold_uncertainty.csv')
    
	
if __name__ == '__main__':
    main()
	