#!/bin/bash

module load singularity

echo "U-Net cross-validation"

singularity exec --nv -w ../../../ubuntu_container_2/ python3 ../examples/main_cross_validation_UNet.py


