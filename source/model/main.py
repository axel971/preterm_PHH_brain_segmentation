import os
import numpy as np
from model import unet3d
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from generator_array import Generator
from dataio import printgpu, import_data_filename, write_nii
from image_process import load_image_correct_oritation, crop_pad3D, resize
#from viewer import view3plane
from one_hot_label import redefine_label
import metrics
import metrics2
import time
import pickle
from prediction_my import evaluate, test
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import pandas
import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
from utils import get_nii_data, get_nii_affine, save_image

def main():
    n_subject = 17  # len(subject_index)
    subject_list = np.arange(n_subject)
    image_size = [256, 156, 256]
    patch_size = [64, 64, 64]
    labels = [0, 1, 2, 3, 4, 5]  # redefine labels

    ###########################################################################
    # read data
    file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.xlsx"
    data = pandas.read_excel(file_dir, index=False)
    data_length = data.shape
    subject_names = np.array(data['subject_name'])
    img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing'
    delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
    model_weight_path = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/evaluation_hyper_parameters/weights/'
 	
    np.random.shuffle(subject_list)
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]),
                 dtype=np.float32)
    Y = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]),
                 dtype=np.float32)
    affines = []
                 
    for n in range(n_subject):
        X[n] = get_nii_data(img_dir + '/' + subject_names[n] + '_preproc.nii.gz' )
        Y[n] = get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '_preproc.nii.gz'))
        
    affines = np.array(affines)
    
    # toc = time.clock()

    printgpu('..data size:' + str(X.shape))
    printgpu('..loading data finished ' + str(time.asctime(time.localtime())))

    ID = 8  # 10-fold
    training_generator = Generator(X[:ID, ...], Y[:ID, ...],
                                   batch_size=4, patch_size=patch_size,
                                   labels=labels, argmentation=False)
    printgpu('..generator_len: ' + str(training_generator.__len__()))
    printgpu('..training_n_subject: ' + str(training_generator.n_subject))
    printgpu('..training_n_patch_per_sub: ' + str(training_generator.loc_patch.n_patch))
    printgpu('..step_by_epoch:' + str(training_generator.step_by_epoch))
#     printgpu('..self.current_epoch:' + str(training_generator.current_epoch))
#     printgpu('..total_step:' + str(training_generator.total_step))
#     printgpu('..n_epoch_all_data:' + str(training_generator.n_epoch_all_data))

    validation_generator = Generator(X[ID:, ...], Y[ID:, ...],
                                     batch_size=12, patch_size=patch_size,
                                     labels=labels, argmentation=False)
    printgpu('..validation_n_subject: ' + str(validation_generator.n_subject))

    #############################################################
    # build model
    single_model = unet3d(patch_size+[1], len(labels))
    model = multi_gpu_model(single_model, gpus=2)
    
    # model.summary()
    # keras.utils.print_summary(model,line_length=120)
    optimizer = optimizers.Adam(lr=1e-4)
    # optimizer = AdamAccumulate(lr=1e-4, accum_iters=10)
    model.compile(optimizer=optimizer,
                  # loss=metrics.jaccard_distance_loss,
                  # loss='binary_crossentropy',
#                   loss='categorical_crossentropy',
                  loss = metrics.focal_loss(),
                  # loss=my_loss_fun,
                  metrics=[metrics.dice_multi])
    check_pointer = ModelCheckpoint(filepath = model_weight_path + 'weights_{epoch:d}.h5', period = 1, save_weights_only = True)
    tensor_board_callback = TensorBoard(log_dir='./log')
    # model.load_weights('mymodel_weights.h5')
    hist = model.fit_generator(generator=training_generator,
                               # steps_per_epoch=10,
                               epochs=30,
                               validation_data=validation_generator,
                               # validation_steps=300,
                               verbose=2,
#                                callbacks=[check_pointer, tensor_board_callback]
                               callbacks=[tensor_board_callback]
                               )
                               
                               
    model.save_weights('mymodel_weights.h5')
    
    with open('history_noflip', 'wb') as file_handle:
        pickle.dump(hist.history, file_handle)


if __name__ == '__main__':
    main()
