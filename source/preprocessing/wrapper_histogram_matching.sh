#!/bin/bash

img_dir=/home/axel/dev/neonatal_brain_segmentation/data/images/case/raw
img_preproc_dir=/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing

subject_id=$1 
subject_0_id=$2
 
echo "Histogram matching starting..."
time ./histogram_matching/build/histogramMatching   ${img_preproc_dir}/${subject_0_id}_preproc.nii.gz  ${img_preproc_dir}/${subject_id}_preproc.nii.gz  ${img_preproc_dir}/${subject_id}_preproc.nii.gz
echo "Histogram matching ending..."