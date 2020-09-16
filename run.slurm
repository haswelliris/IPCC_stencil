#!/bin/bash
#SBATCH -J ipcc
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH --exclusive

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

echo "build..."
make
echo ""
echo "running..."
./bin/stencil IPCC.png