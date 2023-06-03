#!/bin/bash
# Install the MERlin package on Harvard RC
# Run this script in the home directory of MERlin

module load Mamba/4.14.0-0
module load gcc/8.2.0-fasrc01

mamba create --name merlin_env_2 python=3.8
mamba activate merlin_env_2

pip install -e .