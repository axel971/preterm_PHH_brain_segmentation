#!/bin/bash

module load tensorflow
module load keras
module load singularity

singularity exec --nv -w ../../../ubuntu_container/ python3 ../evaluation/show_p_concreteDropout.py
