#!/bin/bash
#SBATCH -J ipcc
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH --exclusive

module load intel/20.0.1-lcc
#module load aocc/2.0.0
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

echo "build..."
make
echo ""
echo "running..."
./bin/stencil IPCC.png