#!/bin/bash

module load tensorflow
module load singularity
echo 'denseNet'

singularity exec --nv -w ../../../ubuntu_container_2/ python3 ../examples/main_cross_validation_denseNet.py


