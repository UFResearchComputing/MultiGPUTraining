# Distributed Data Parallel Training with Multi-GPU on HiPerGator-AI

- Yunchao Yang
- UF Research Computing

The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.
Code is forked and adapted for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html


## How to use this Repo.

Step 0: navigate to `ood.rc.ufl.edu` and request a open jupyter notebook with 1 node and 2 GPUs:  
  - Cluster partition = "gpu"
  - Generic Resource Request = "gpu:geforece:2" or "gpu:a100:2"
  - Additional SLURM Options = --reservation=?? (only used during the workshop period)

## Get started with the starter code
* [single_gpu.py](single_gpu.py): Non-distributed training script on a single GPU
* [How to run]: `python single_gpu.py` 

## Exercise 1: adapt your serial code to single node multiple processes run with `mp.spawn` utility and user-specified setting. 
* [exercise1_multigpu.py](multigpu.py): DDP on a single node, with `mp.spawn`  
* [exercise1_run_multigpu.sh](run_multigpu.sh): runner code
You can test your code by run `./exercise1_run_multigpu.sh`

## Exercise 2: adapt your serial code to single node multiple processes using the torchrun utility
* [exercise2_multigpu_torchrun.py](multigpu.py): DDP setup on a single node using `torchrun`
* [exercise2_run_multigpu_torchrun.sh](run_multigpu_torchrun.sh): runner
You can test your code by run `exercise2_run_multigpu_torchrun.sh`

## Step 3. Run MultiNode parallel jobs using SLURM  on HPG (work offline)
* [slurm-moltinode/multigpu_torchrun.py](slurm/multigpu_torchrun.py): training script for multiGPU
* [slurm-moltinode/launch_ddp_2N4G.sh](slurm/launch_ddp_2N4G.sh): Sample slurm script to launch a trining script using torchrun on 2 Nodes with 2 GPUs on each node.
* [slurm-moltinode/launch_ddp_4N4G.sh](slurm/launch_ddp_4N4G.sh): Sample slurm script to launch a trining script using torchrun on 4 Nodes with 1 GPUs on each node.
You can submit your SLURM job script by run `sbatch launch_ddp_2N4G.sh`

## Solutions
You can find the solution to exercises in this folder.

## Learn more about the detailed code walkthrough
Please follow the [Distributed Data Parallel in PyTorch Tutorial Series](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj).


# License
MIT 