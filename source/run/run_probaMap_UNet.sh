#!/bin/bash

module load tensorflow
module load singularity

singularity exec --nv -w ../../ubuntu_container/ python3 ./model/main_cross_validation_UNet_probaMap.py
