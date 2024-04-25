#!/bin/bash

module load tensorflow
module load singularity

echo "U-Net"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_training_UNet.py


