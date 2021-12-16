# fdelmodc2D-gpu
This is a GPU version for Jan Thorbecke's `fdelmodc`. It only supports 2D acoustic 
modeling. It uses `slurm` to divide work among nodes and `gnu parallel` to divide 
work among GPUs inside a node. Workload division is among shots only; no domain 
decomposition is applied (please keep GPU maximum memory in mind when using).
Victor Koehne
SENAI-CIMATEC, 2020

# Installation and compilation
Put the folder `fdelmodc2D-gpu` on your `Opensource` directory (side by side with 
all the other folders)
To install:
```sh
# In a GPU node
sh make.sh install
# From login node
srun -p GPUPARTITION sh make.sh install
```
If all went well, `ls ../bin` should show the executables:
```sh
fdelmodc2D_singleGPU
parallel
```
The only relevant executables are `fdelmodc2D_singleGPU` and `parallel`, please do not use the others as they may be unstable.

# Uninstalling 
`sh make.sh uninstall` should remove all `.o` and executables from the folder. Please also check `ls ../bin` and see if the executables were removed.

# Testing

Check folder `demo/modeling_leon_diekmann` .

