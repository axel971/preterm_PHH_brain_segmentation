#!/bin/bash

#module load tensorflow
module load singularity

singularity exec --nv -w ../../../ubuntu_container_2/  python3 ../examples/main_get_prediction_cross_validation_U-Net.py
