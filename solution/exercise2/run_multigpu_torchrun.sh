#! /bin/bash

# run this DDP code with multiGPUs on a single Node
# 1. request resources from interactive mode or OOD
 
module load pytorch/1.10

torchrun --standalone --nproc_per_node=2 multigpu_torchrun.py 50 10
