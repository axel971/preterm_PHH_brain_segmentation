
# Get paths for each files
#path_data1 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_01_exp2/volumeRelativeError/volumeRelativeError.csv'
#path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/denseNet/exp1/volumeRelativeError/volumeRelativeError.csv'
#path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/volumeRelativeError/volumeRelativeError.csv'
#path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/volumeRelativeError/volumeRelativeError.csv'
#path_data2 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp1/volumeRelativeError/volumeRelativeError.csv'

path_data1 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp1/volumeRelativeError/volumeRelativeError.csv'
#path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/denseNet/exp1/volumeRelativeError/volumeRelativeError.csv'
#path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/volumeRelativeError/volumeRelativeError.csv'
#path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/volumeRelativeError/volumeRelativeError.csv'
path_data2 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_01_exp2/volumeRelativeError/volumeRelativeError.csv'



# Open the files
data1 = read.csv(path_data1, header = T)
data2 = read.csv(path_data2, header = T)

data1 = data1[1:(nrow(data1)-3),2:ncol(data1)]
data2 = data2[1:(nrow(data2)-3),2:ncol(data2)]

# Compute the wilxocon test
#print(wilcox.test(as.numeric(data1[ , 2]), as.numeric(data2[ , 2]), exact = T, paired = T, alternative = "greater"))
print(wilcox.test(as.numeric(data1[ , 7]), as.numeric(data2[, 7]), exact = T, paired = T, alternative = "less"))