import numpy as np
from model.image_process import crop3D_hotEncoding
from tensorflow.keras import backend as K
import metrics.metrics as metrics
import metrics.metrics2 as metrics2
from model.one_hot_label import restore_labels

# Voxel-wise uncertainty

def predictive_entropy(MCsamples, image_size, labels):

	#Normalization of the MC samples (each voxel must be probability values comprise between 0 and 1)	
	MCsamples_normalized = MCsamples / np.sum(MCsamples, axis = -1, keepdims = True)
	
	
	#Computation of the entropy from the normalized MC samples
	mean = np.mean(MCsamples_normalized, axis = 0)
	mean = crop3D_hotEncoding(mean, image_size, len(labels))

	eps = K.epsilon()
	entropy = np.multiply(mean, np.log(mean + eps))
	entropy = -1 * np.sum(entropy, axis = -1)

	
	return entropy	
	
def mutual_information(MCsamples, image_size, labels):
	
	H = predictive_entropy(MCsamples, image_size, labels)
	
	eps = K.epsilon()
	tmp = np.sum(np.multiply(MCsamples, np.log(MCsamples + eps)), axis = 0)
	tmp = crop3D_hotEncoding(tmp, image_size, len(labels))
	tmp = np.sum(tmp, axis = -1)
	
	
		
	return H + (tmp / np.shape(MCsamples)[0])
	
	
# Structure wise uncertainty

def dice_uncertainty(MCsamples, labels):

	dices = []
	mean = np.mean(MCsamples, axis = 0)
	mean = restore_labels(mean, labels)
	
	for iSample in range(MCsamples.shape[0]):
		dices.append(metrics.dice_multi_array(mean, restore_labels(MCsamples[iSample], labels), labels))
	
	dices = np.array(dices)
	diceMean = np.mean(dices, axis = 0)
	diceSD = np.std(dices, axis = 0)
	
	return diceSD


def hd95_uncertainty(MCsamples, labels, voxelspacing):

	hd95s = []
	mean = np.mean(MCsamples, axis = 0)
	mean = restore_labels(mean, labels)
	
	for iSample in range(MCsamples.shape[0]):
		hd95s.append(metrics2.hd95_multi_array(mean, restore_labels(MCsamples[iSample], labels), labels, voxelspacing))
	
	hd95s = np.array(hd95s)
	hd95Mean = np.mean(hd95s, axis = 0)
	hd95SD = np.std(hd95s, axis = 0)
	
	return hd95Mean


def assd_uncertainty(MCsamples, labels, voxelspacing):

	assds = []
	mean = np.mean(MCsamples, axis = 0)
	mean = restore_labels(mean, labels)
	
	for iSample in range(MCsamples.shape[0]):
		assds.append(metrics2.assd_multi_array(mean, restore_labels(MCsamples[iSample], labels), labels, voxelspacing))
	
	assds = np.array(assds)
	assdMean = np.mean(assds, axis = 0)
	assdSD = np.std(assds, axis = 0)
	
	return assdMean
	
def volume_uncertainty(MCsamples, labels, voxelspacing):

	volumes = []
	
	for iSample in range(MCsamples.shape[0]):
	
		# Labeled the current MCsample
		currentLabeledMCSample = restore_labels(MCsamples[iSample], labels)
		
		# Initialize a vector that will stock the volume of each delineation
		currentVolumes = np.zeros(np.shape(labels), dtype = np.float32) 
	
		# Compute the volume of each delineation wrt their labels
		for iLabel in range(len(labels)):
  	  		binaryMap = (currentLabeledMCSample == labels[iLabel])
  	  		currentVolumes[iLabel] = np.sum(binaryMap)

		# Stock the delineation volumes obtained from the current MCsample		
		volumes.append(currentVolumes)

	# Compute the voxel volume of the MCsamples
	voxelVolume = 1.
	for iSpacing in range(len(voxelspacing)):
		voxelVolume *= voxelspacing[iSpacing]
	
	volumes = np.array(volumes) * voxelVolume
	volumeMean = np.mean(volumes, axis = 0)
	volumeSD = np.std(volumes, axis = 0)
	
	return volumeSD/volumeMean


def MAE_uncertaintyMaps(uncertainty1, uncertainty2):
	
	return np.mean(np.abs(uncertainty1 - uncertainty2))