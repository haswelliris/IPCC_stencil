#!/bin/bash
#SBATCH -J ipcc
#SBATCH -p amd_256
#SBATCH -N 2
#SBATCH --exclusive

module load intel/17.0.7-thc-public3
module load mpi/openmpi/3.1.4-icc17-cjj-public3
#module load aocc/2.0.0
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
export I_MPI_FABRICS=shm:ofa

echo "build..."
make
echo "built."

echo ""
echo "running..."
srun -p amd_256 -N 2 -n 2 -c 64 bin/stencil IPCC.png