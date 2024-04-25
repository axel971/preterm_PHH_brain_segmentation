

weight='/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/evaluation_hyper_parameters/weights/weights_1.h5'
output_img_dir='/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/evaluation_hyper_parameters/prediction'

mkdir ${output_img_dir}/epoch_1/

python3 test.py ${output_img_dir}/epoch_1/ ${weight}