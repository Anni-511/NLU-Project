#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=nlu
#SBATCH --output=./out/val_%j.out                                                                                                                                                                     
#SBATCH --error=./err/val_%j.err

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1
module purge
set -x

singularity exec --nv \
	    --overlay /scratch/rv2138/zillowcapstone/overlay.ext3:ro \
	    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif  \
	    /bin/bash -c "source /ext3/env.sh; \
					  conda activate; \
					  python -m train.train \
					  --slurm ${SLURM_JOB_ID}"