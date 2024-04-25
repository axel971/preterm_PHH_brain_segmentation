#!/bin/bash

pathInputFolder='/home/axel/dev/neonatal_brain_segmentation/data/MRI_segmentation'
Home='/home/axel/dev/neonatal_brain_segmentation'
pathOutputFolder=${Home}'/data/delineations/case/raw'

# Build the repositories
mkdir ${Home}/data
mkdir ${Home}/data/delineations
mkdir ${Home}/data/delineations/case/

# Get the raw data
for subjectFolder in ${pathInputFolder}/*  #Go toward each subject folder
do 
    #Get subject name
	subjectName=$(basename "$subjectFolder")
	
	#Copy the patient image in the output folder
	cp  ${pathInputFolder}/${subjectName}/scan_01/files/segmentations/${subjectName}_*indeterminate.nii.gz ${pathOutputFolder}/${subjectName}_scan_01_delineations.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_02/files/segmentations/${subjectName}_*indeterminate.nii.gz ${pathOutputFolder}/${subjectName}_scan_02_delineations.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_03/files/segmentations/${subjectName}_*indeterminate.nii.gz ${pathOutputFolder}/${subjectName}_scan_03_delineations.nii.gz
	
	cp  ${pathInputFolder}/${subjectName}/scan_01/files/recon/segmentations/${subjectName}_*labels_final.nii.gz ${pathOutputFolder}/${subjectName}_scan_01_delineations.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_02/files/recon/segmentations/${subjectName}_*labels_final.nii.gz ${pathOutputFolder}/${subjectName}_scan_02_delineations.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_03/files/recon/segmentations/${subjectName}_*labels_final.nii.gz ${pathOutputFolder}/${subjectName}_scan_03_delineations.nii.gz
	
	cp  ${pathInputFolder}/${subjectName}/scan_01/files/segmentations/${subjectName}_*labels_final.nii.gz ${pathOutputFolder}/${subjectName}_scan_01_delineations.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_02/files/segmentations/${subjectName}_*labels_final.nii.gz ${pathOutputFolder}/${subjectName}_scan_02_delineations.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_03/files/segmentations/${subjectName}_*labels_final.nii.gz ${pathOutputFolder}/${subjectName}_scan_03_delineations.nii.gz
	
done	