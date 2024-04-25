#!/bin/bash

module load tensorflow
module load singularity

#echo "Date: 12/28/20"
echo "run_bayesian_U-Net"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_training_Bayesian_UNet.py
