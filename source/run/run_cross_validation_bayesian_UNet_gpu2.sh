#!/bin/bash

module load tensorflow
module load singularity

#echo "Date: 12/28/20"
echo "run_bayesian_U-Net"

singularity exec --nv -w ../../../ubuntu_container_2/ python3 ../examples/main_cross_validation_Bayesian_UNet_exp2.py
