
# Get paths for each files
path_data1 = '/home/axel/dev/neonatal_brain_segmentation/data/output/denseNet/exp1/dice/dice.csv'
#path_data1 = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/dice/dice.csv'
#path_data1 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/ensembleLearning/dice/dice.csv'
#path_data1 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_encoder_decoder/p_01_exp2/dice/dice.csv'
#path_data1 =  '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp1/dice/dice.csv'

# Open the files
data1 = read.csv(path_data1, header = T)
data1 = data1[1:(nrow(data1)-3),2:ncol(data1)]

# Compute histogramm


tiff("/home/axel/dev/neonatal_brain_segmentation/screenshot/cumulative_histogramm/cumulative_histogram_dice_score_brainsterm_DenseNet.tiff", width = 12, height = 10, units = 'in', res = 500)
h = hist(data1[, 7], main = "", breaks = c(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0), xlim=c(0,1), ylim = c(0,36))
h$counts = cumsum(h$count)
plot(h, , ylab= "Number of subjects", xlab = "Dice score values for cerebellum (DenseNet)", xaxt = "n",yaxt = "n", main = "")
axis(1, at=seq(0.0, 1.0, by=0.05), labels=sprintf("%.2f",seq(0.0,1.0, by = 0.05)), cex.axis = 1.5)
axis(2, at=seq(0, 36, by= 1), labels=seq(0, 36, by = 1), las = 1, cex.axis = 1.5)
dev.off()