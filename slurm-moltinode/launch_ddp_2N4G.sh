#!/bin/bash

# SLURM Script to launch a multi-node multi-GPU pytorch DDP training 
# on UF HiPerGator's AI partition  
# 11/2022, Yunchao Yang, UF Research Computing
# The python script is adapted from pytorch ddp tutorial 

# Torchrun on multiNode multiGPU, 
#       set #SBATCH --ntasks=--nodes
#       set #SBATCH --ntasks-per-node=1  
#       set #SBATCH --gpus=total number of processes to run on all nodes
#       set #SBATCH --gpus-per-task=--gpus / --ntasks  
########################################################################################

# In this example, we are requesting 2 nodes with 2 GPUs per node, total GPUs = 4.

#SBATCH --nodes=2               # How many DGX nodes? Each has 8 A100 GPUs
#SBATCH --ntasks=2              # How many tasks? One per GPU
#SBATCH --ntasks-per-node=1     # 1 torchrun per node
#SBATCH --gpus-per-task=2       # #GPU per srun step task

#SBATCH --cpus-per-task=4       # How many CPU cores per task, equals to number of dataloader worker
#SBATCH --mem=24gb             # host memory per node--up to 2TB for hpg-ai (Not GPU memory--that is 80GB per A100 GPU)
#SBATCH --partition=hpg-ai      # Specify the HPG AI partition

# Enable the following to limit the allocation to a single SU.
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out


module load pytorch/1.10
#or using conda environment
# module load conda
# conda activate your_env_name

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
echo Node list $nodes_array

#head_node=${nodes_array[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
#echo Node IP: $head_node_ip

# if above head_node_ip threw an error with hostname, use the following simple way to extract ip-address 
head_node_ip=`hostname --ip-address`
echo HeadNodeIP: $head_node_ip

head_node_port=29500

export LOGLEVEL=INFO
# For NCCL debugging
# export NCCL_DEBUG=WARN #change to INFO if debugging DDP

pwd; hostname; date

srun --export=ALL torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:$head_node_port \
multigpu_torchrun.py 50 10

