#!/bin/bash

module load slurm
list=/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.csv
ID=''

for subject_name in `cat $list`
do
	subject_name=`echo $subject_name | cut -d ',' -f2`
	
	if [[ $subject_name != "subject_name" ]]
    then
    	tmp=`cat $list`
		tmp=(${tmp})	
		subject_0_name=`echo ${tmp[1]} | cut -d ',' -f2`
		tmp_ID=`sbatch --output=./slurmlogs/${subject_name}_preproc.out ./wrapper_preproc.sh ${subject_name} ${subject_0_name}`
		tmp_ID=`echo $tmp_ID | cut -d ' ' -f4`
		ID=$ID':'$tmp_ID
   fi
done

echo $ID

# 
for subject_name in `cat $list`
do
	subject_name=`echo $subject_name | cut -d ',' -f2`
	
	if [[ $subject_name != "subject_name" ]]
    then
    	tmp=`cat $list`
		tmp=(${tmp})	
		subject_0_name=`echo ${tmp[1]} | cut -d ',' -f2`
		sbatch --dependency=afterok$ID --output=./slurmlogs/${subject_name}_histogram_matching.out ./wrapper_histogram_matching.sh ${subject_name} ${subject_0_name}
   fi


done

