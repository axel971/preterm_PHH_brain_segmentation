
# Get paths for each files
#path_data1 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_01_exp1/AUC_uncertainty/uncertainty_AUC.csv'
#path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_01_exp2/AUC_uncertainty/uncertainty_AUC.csv'
#path_data3 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_01_exp3/AUC_uncertainty/uncertainty_AUC.csv'

path_data1 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp1/AUC_uncertainty/uncertainty_AUC.csv'
path_data2 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp2/AUC_uncertainty/uncertainty_AUC.csv'
path_data3 = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp3/AUC_uncertainty/uncertainty_AUC.csv'


# Open the files
data1 = read.csv(path_data1, header = T)
data2 = read.csv(path_data2, header = T)
data3 = read.csv(path_data3, header = T)

data1 = data1[1:(nrow(data1)-3),2:ncol(data1)]
data2 = data2[1:(nrow(data2)-3),2:ncol(data2)]
data3 = data3[1:(nrow(data3)-3),2:ncol(data3)]

# Combine the data in one matrix

#data = cbind(data1[ , 7], data2[ , 7], data3[ , 7])
data = cbind(data1, data2, data3)


# Compute the friedman test
print(friedman.test(data), exact = T, paired = T)


#print(wilcox.test(as.numeric(data1[ , 7]), as.numeric(data2[, 7]), exact = T, paired = T, alternative = "less"))