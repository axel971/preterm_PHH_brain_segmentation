import os
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
import numpy as np
from image_process import crop_edge_pair, load_image_correct_oritation
from dataio import import_data_filename, write_nii
from one_hot_label import redefine_label
import metrics
import time
import pickle
import gc
import uncertainty

# from viewer import view3plane
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import pandas
import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
from utils import get_nii_data, get_nii_affine, save_image, get_voxel_spacing


def main():
	
	n_subject = 36
	file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
	data = pandas.read_excel(file_dir, index=False)
	data_length = data.shape
	subject_names = np.array(data['subject_name'])
    
	MCsamples_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_decoder/p_05/MCsamples/allVOI_cross_entropy_loss'
	image_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing' 
	hd95_uncertainty_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_decoder/p_05/hd95_uncertainty/hd95_uncertainty.csv'
	image_size = [256, 128, 256]
	patch_size = [64, 64, 64]
	labels = [0, 1, 2, 3, 4, 5, 6]
	
	hd95_uncertainty = []
	
	for iSubject in range(n_subject):
	
		affine = get_nii_affine(image_dir + '/' + subject_names[iSubject] + '_preproc.nii.gz')
		voxel_spacing = get_voxel_spacing(image_dir + '/' + subject_names[iSubject] + '_preproc.nii.gz')
		MCsamples = np.load(MCsamples_dir + '/' + subject_names[iSubject] + '_MCsamples.npy')
		
		hd95_uncertainty.append(uncertainty.hd95_uncertainty(MCsamples, labels, voxel_spacing))

	
	data = pandas.DataFrame(hd95_uncertainty)	
	data.to_csv(hd95_uncertainty_dir)
    	
		
if __name__ == '__main__':
    main()
