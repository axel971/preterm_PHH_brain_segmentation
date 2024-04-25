#!/bin/bash

module load tensorflow
module load singularity

echo 'bayesian U-Net with Spatial3DConcreteDropout'

singularity exec --nv -w ../../../ubuntu_container_2/ python3 ../examples/main_cross_validation_Bayesian_UNet_Spatial3DConcreteDropout.py
