#!/bin/bash

module load tensorflow
module load singularity

singularity exec --nv -w ../../ubuntu_container/ python3 ./3D_U-Net/main_cross_validation_autoContext_Bayesian_UNet.py
