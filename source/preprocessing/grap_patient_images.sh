#!/bin/bash

pathInputFolder='/home/axel/dev/neonatal_brain_segmentation/data/MRI_segmentation'
Home='/home/axel/dev/neonatal_brain_segmentation'
pathOutputFolder=${Home}'/data/images/case/raw'

# Build the repositories
#mkdir ${Home}/data
#mkdir ${Home}/data/images
#mkdir ${Home}/data/images/case

# Get the raw data
for subjectFolder in ${pathInputFolder}/*  #Go toward each subject folder
do 
    #Get subject name
	subjectName=$(basename "$subjectFolder")
	
	#Copy the patient image in the output folder
	cp  ${pathInputFolder}/${subjectName}/scan_01/files/T2/${subjectName}_*.nii.gz ${pathOutputFolder}/${subjectName}_scan_01.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_02/files/T2/${subjectName}_*.nii.gz ${pathOutputFolder}/${subjectName}_scan_02.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_03/files/T2/${subjectName}_*.nii.gz ${pathOutputFolder}/${subjectName}_scan_03.nii.gz
	
	cp  ${pathInputFolder}/${subjectName}/scan_01/files/recon/T2/${subjectName}_*.nii.gz ${pathOutputFolder}/${subjectName}_scan_01.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_02/files/recon/T2/${subjectName}_*.nii.gz ${pathOutputFolder}/${subjectName}_scan_02.nii.gz
	cp  ${pathInputFolder}/${subjectName}/scan_03/files/recon/T2/${subjectName}_*.nii.gz ${pathOutputFolder}/${subjectName}_scan_03.nii.gz
done	