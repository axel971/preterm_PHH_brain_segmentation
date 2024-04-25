#!/bin/bash

delineation_dir=/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/raw
delineation_relabeled_dir=/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/relabeled
delineation_preproc_allVOIs_dir=/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs
delineation_preproc_ventricle_dir=/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/ventricle
img_dir=/home/axel/dev/neonatal_brain_segmentation/data/images/case/raw
img_preproc_dir=/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing

subject_id=$1 
subject_0_id=$2
 
echo "Starting relabelling..."
 time ./relabeling/build/relabeling ${delineation_dir}/${subject_id}_delineations.nii.gz ${delineation_relabeled_dir}/${subject_id}_delineations.nii.gz 
echo "Ending relabelling..."

echo "N4 Bias field correction starting..."
 time /cm/shared/apps/mirtk/Packages/DrawEM/ThirdParty/ITK/N4 -s '2' -c [5x5x5,0.001] -i ${img_dir}/${subject_id}.nii.gz -x ${delineation_relabeled_dir}/${subject_id}_delineations.nii.gz -o ${img_preproc_dir}/${subject_id}_preproc.nii.gz
echo "N4 Bias field correction ending..."

echo "Image Resampling starting..."
time ./resampling/build/resampling  ${img_preproc_dir}/${subject_id}_preproc.nii.gz  ${img_preproc_dir}/${subject_id}_preproc.nii.gz
echo "Image Resampling ending..."

echo "Delineation Resampling starting..."
time ./resampling_delineation/build/resampling_delineation  ${delineation_relabeled_dir}/${subject_id}_delineations.nii.gz  ${delineation_preproc_allVOIs_dir}/${subject_id}_delineations.nii.gz
echo "Delineation Resampling ending..."

#echo "Starting label injury..."
#time ./label_injury/build/label_injury ${delineation_preproc_allVOIs_dir}/${subject_id}_delineations.nii.gz ${delineation_preproc_allVOIs_dir}/${subject_id}_delineations.nii.gz
#echo "Ending label injury..."

#echo "Get ventricle delineations starting..."
#time ./get_ventricle_delineation/build/get_ventricle_delineation  ${delineation_preproc_allVOIs_dir}/${subject_id}_delineations.nii.gz  ${delineation_preproc_ventricle_dir}/${subject_id}_delineations.nii.gz
#echo "Get ventricle delineation ending..."

# echo "Histogram matching starting..."
# time ./histogram_matching/build/histogramMatching   ${img_preproc_dir}/${subject_0_id}_preproc.nii.gz  ${img_preproc_dir}/${subject_id}_preproc.nii.gz  ${img_preproc_dir}/${subject_id}_preproc.nii.gz
# echo "Histogram matching ending..."