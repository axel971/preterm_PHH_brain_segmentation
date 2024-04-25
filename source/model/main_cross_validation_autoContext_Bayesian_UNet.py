import os
import tensorflow as tf
from model import bayesian_Unet3d_autoContext
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from generator_array import Generator_with_two_modality
import numpy as np
from prediction_my import Bayesian_autoContext_Predict
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
    n_subject = 18
    subject_list = np.arange(n_subject)
    image_size = [256, 128, 256]
    patch_size = [64, 64, 64]
    labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels
    # np.random.shuffle(subject_list)
    
    # Get path data
    file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.xlsx"
    uncertainty_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/predictive_entropy/allVOI_cross_entropy_loss' 
    data = pandas.read_excel(file_dir, index=False)
    data_length = data.shape
    subject_names = np.array(data['subject_name'])
    img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing'
    delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
    
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    uncertainty = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    Y = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    affines = []
    
    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '_preproc.nii.gz' )
        Y[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz' )
        uncertainty[n] = get_nii_data(uncertainty_dir + '/' + subject_names[n] + '_predictiveEntropy.nii' )
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
    	
    	single_model = bayesian_Unet3d_autoContext(patch_size + [1], patch_size + [1], len(labels))
    	model = multi_gpu_model(single_model, 2)
    	optimizer = optimizers.Adam(lr = 1e-3)
    	


    	training_generator = Generator_with_two_modality(X[train_id, ...], uncertainty[train_id, ...], Y[train_id, ...], batch_size = 2, patch_size=patch_size, labels=labels)
    	
    	print('..generator_len: ' + str(training_generator.__len__()), flush=True)
    	print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
    	print('..training_n_patch_per_sub: ' + str(training_generator.loc_patch.n_patch), flush=True)

    	model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=[metrics.dice_multi])
    	model.fit_generator(generator=training_generator, epochs = 60, verbose = 2)
    	#model.save_weights('mymodel_weights_'+str(iFold)+'.h5')
    	
    	##### Prediction ####
    	    	
    	#with tf.device('/cpu:0'):
    #	prediction, probaMaps = bayesian_test(X[test_id], model, image_size, patch_size, labels)
    
    	
    	predictor = Bayesian_autoContext_Predict(model, image_size, patch_size, labels)
    	
    	for i in range(len(test_id)):
    	
    		prediction, MCsamples = predictor.__run__(X[test_id[i]])
    		
    		dice.append(metrics.dice_multi_array(Y[test_id[i]], prediction, labels))
    		np.save('/home/axel/dev/neonatal_brain_segmentation/data/output/autocontext_bayesian_UNet/MCsamples/allVOI_cross_entropy_loss/' + subject_names[test_id[i]] + '_MCsamples.npy', MCsamples)
    		save_image(prediction, affines[test_id[i]], '/home/axel/dev/neonatal_brain_segmentation/data/output/autocontext_bayesian_UNet/prediction/allVOI_cross_entropy_losss/' + subject_names[test_id[i]] + '_predicted_segmentation.nii')
    	
    	model.save('/home/axel/dev/neonatal_brain_segmentation/data/output/autocontext_bayesian_UNet/models/model_' + str(iFold) + '.h5')
		
    	del single_model
    	del training_generator
    	
    	K.clear_session()
    	gc.collect()

    print(dice)
    print(np.mean(dice, 0))
    
if __name__ == '__main__':
    main()
