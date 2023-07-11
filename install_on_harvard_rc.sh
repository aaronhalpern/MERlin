#!/bin/bash
# Install the MERlin package on Harvard RC
# Run this script in the directory containing MERlin

module load python/3.10.9-fasrc01
module load gcc/12.2.0-fasrc01

mamba create --name merlin_update
source activate merlin_update

pip install -e MERlin