#!/bin/bash

module load tensorflow
module load singularity

singularity exec --nv -w ../../../ubuntu_container/  python3 ../examples/main_prediction_Bayesian_UNet_test.py
