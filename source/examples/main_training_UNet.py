import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
import os
from model.model import unet3d
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator
import numpy as np
from model.prediction_my import evaluate, test
from model.image_process import crop_edge_pair, load_image_correct_oritation
from model.dataio import import_data_filename, write_nii
from model.one_hot_label import redefine_label
import metrics.metrics as metrics
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
   
   
    print('..data size:' + str(X.shape), flush=True)
    print('..loading data finished ' + str(time.asctime(time.localtime())), flush=True)
    
    
    	
    single_model = unet3d(patch_size+[1], len(labels))
    model = multi_gpu_model(single_model,2)
    optimizer = optimizers.Adam(lr=1e-3)
    	

    training_generator = Generator(X, Y, batch_size = 4, patch_size=patch_size, labels=labels)
    	
    print('..generator_len: ' + str(training_generator.__len__()), flush=True)
    print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
    print('..training_n_patch_per_sub: ' + str(training_generator.loc_patch.n_patch), flush=True)

    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=[metrics.dice_multi])
    	
    start_training_time = time.time()	
    model.fit_generator(generator=training_generator, epochs=60, verbose=2)
    end_training_time = time.time()	
    print('training time: ' + str(end_training_time - start_training_time))

    model.save('/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/models/model_training_whole_cohort.h5')

    
if __name__ == '__main__':
    main()
