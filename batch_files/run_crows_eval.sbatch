#!/bin/bash

# We assume running this from the script directory
job_directory=$PWD/batch_files/
data_dir="${SCRATCH}/project/LizardLips"

Checkpoints=('/scratch/rv2138/nlu/checkpoints/19114448/checkpoint-15000/' '/scratch/rv2138/nlu/checkpoints/19114459/checkpoint-30000/' '/scratch/rv2138/nlu/checkpoints/19114460/checkpoint-75000/' '/scratch/rv2138/nlu/checkpoints/19114476/checkpoint-151000/' '/scratch/rv2138/nlu/checkpoints/19114515/checkpoint-1000/' '/scratch/rv2138/nlu/checkpoints/19114525/checkpoint-7000/' '/scratch/rv2138/nlu/checkpoints/19114529/checkpoint-15000/' '/scratch/rv2138/nlu/checkpoints/19114530/checkpoint-30000/' '/scratch/rv2138/nlu/checkpoints/19114557/checkpoint-75000/' '/scratch/rv2138/nlu/checkpoints/19114562/checkpoint-151000/' '/scratch/rv2138/nlu/checkpoints/19109713/checkpoint-1000/' '/scratch/rv2138/nlu/checkpoints/19109719/checkpoint-7000/' '/scratch/rv2138/nlu/checkpoints/19109723/checkpoint-15000/' '/scratch/rv2138/nlu/checkpoints/19109724/checkpoint-30000/' '/scratch/rv2138/nlu/checkpoints/19109726/checkpoint-75000/' '/scratch/rv2138/nlu/checkpoints/19109731/checkpoint-151000/' '/scratch/rv2138/nlu/checkpoints/19190987/checkpoint-7000/' '/scratch/rv2138/nlu/checkpoints/19190988/checkpoint-15000/' '/scratch/rv2138/nlu/checkpoints/19190989/checkpoint-30000/' '/scratch/rv2138/nlu/checkpoints/19190990/checkpoint-50000/' '/scratch/rv2138/nlu/checkpoints/19190992/checkpoint-44000/')
Lmods=('bert' 'bert' 'bert' 'bert' 'distilbert' 'distilbert' 'distilbert' 'distilbert' 'distilbert' 'distilbert' 'roberta' 'roberta' 'roberta' 'roberta' 'roberta' 'roberta' 'deberta-v3' 'deberta-v3' 'deberta-v3' 'deberta-v3' 'deberta-v3')
batchfile=('bert10.out' 'bert20.out' 'bert50.out' 'bert100.out' 'distilbert1.out' 'distilbert5.out' 'distilbert10.out' 'distilbert20.out' 'distilbert50.out' 'distilbert100.out' 'roberta1.out' 'roberta5.out' 'roberta10.out' 'roberta20.out' 'roberta50.out' 'roberta100.out' 'deberta-v35.out' 'deberta-v310.out' 'deberta-v320.out' 'deberta-v350.out' 'deberta-v3100.out')

for i in ${!Checkpoints[@]}; do

	CHECKPOINT_NLU=${Checkpoints[i]}
    LM_NLU=${Lmods[i]}
    OUTFILE_NAME_NLU=${batchfile[i]}
    export CHECKPOINT_NLU
    export LM_NLU
    export OUTFILE_NAME_NLU
	sbatch $PWD/batch_files/run_eval.sbatch

done