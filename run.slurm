#!/bin/bash
#SBATCH -J ipcc
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH --exclusive

source /public1/soft/modules/module.sh
module load gcc/8.3.0
module load intel/20.4.3
 
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
./setup-omp.sh
echo "build..."
make
echo ""
echo "running..."
./bin/stencil IPCC.png
#./bin/stencil test-image.png
