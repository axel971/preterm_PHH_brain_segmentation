import os
import tensorflow as tf
from model import unet3d
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from generator_array import Generator
import numpy as np
from prediction_my import predict2
from image_process import crop_edge_pair, load_image_correct_oritation
from dataio import import_data_filename, write_nii
from one_hot_label import redefine_label
import metrics
import time
import pickle
import gc
from sklearn.model_selection import KFold
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
from utils import get_nii_data, get_nii_affine, save_image
from imgaug import augmenters as iaa

def main():
    n_subject = 36
    subject_list = np.arange(n_subject)
    image_size = [256, 128, 256]
    patch_size = [64, 64, 64]
    labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels
    # np.random.shuffle(subject_list)
    
    # Get path data
    file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
    data = pandas.read_excel(file_dir, index=False)
    data_length = data.shape
    subject_names = np.array(data['subject_name'])
    img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing'
    delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
    
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    Y = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    affines = []
    
    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '_preproc.nii.gz' )
        Y[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '_preproc.nii.gz'))
    toc = time.clock()

    print('..data size:' + str(X.shape), flush=True)
    print('..loading data finished ' + str(time.asctime(time.localtime())), flush=True)
    
    
    nFold = 2
    foldSize = int(n_subject/nFold)
    
    dice = []  
    
    for iFold in range(nFold):
    	
    	# Split the data i training and the validation sets for the current fold
    	if iFold < nFold - 1 :
    		test_id = np.array(range(iFold * foldSize, (iFold + 1) * foldSize))
    	else:
    		test_id = np.array(range(iFold * foldSize, n_subject)) # Manage the case where the number of subject is not a multiple of the number of fold
    
    	train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
    	
    	print('---------fold--'+str(iFold + 1)+'---------')
    	print('train: ', train_id)
    	print('test: ', test_id)
    	
    	model = tf.keras.models.load_model('/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/models/model_' + str(iFold) + '.h5', custom_objects = {'dice_multi': metrics.dice_multi})    	
    	    	
    	predictor = predict2(model, image_size, patch_size, labels)
    	
    	for i in range(len(test_id)):
    	
    		prediction = predictor.__run__(X[test_id[i]])
    		os.mkdir('/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/probaMap/' + subject_names[test_id[i]])
    		
    		for iLabel in range(len(labels)):
    			save_image(prediction[:,:, :, iLabel], affines[test_id[i]], '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/probaMap/' + subject_names[test_id[i]] + '/' +  subject_names[test_id[i]] + '_' + str(iLabel) + '_probaMap.nii')
    	
  
    	del model
    	
    	K.clear_session()
    	gc.collect()

    print(dice)
    print(np.mean(dice, 0))
    
if __name__ == '__main__':
    main()
