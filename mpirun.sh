export I_MPI_FABRICS=shm:ofa
module load intel/17.0.7-thc-public3
module load mpi/openmpi/3.1.4-icc17-cjj-public3

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
echo ""
echo "running..."
srun -p amd_256 -N 2 -n 2 -c 64 bin/stencil IPCC.png