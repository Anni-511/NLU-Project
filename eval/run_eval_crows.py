# Import command line arguments and helper functions
import sys
import random
import getpass
from datetime import datetime
import csv
import argparse


from itertools import product
import itertools
import time
import os
import glob


# def dic_element(**args):
#     return args

    
def main():

    ls_lm = ["bert","bert","bert", "bert", "distilbert", "distilbert", "distilbert", "distilbert", "distilbert", "distilbert", "roberta", "roberta", "roberta", "roberta", "roberta", "roberta", "deberta-v3", "deberta-v3", "deberta-v3", "deberta-v3", "deberta-v3"]
    training_id = [19114448,19114459,19114460,19114476,19114515,19114525,19114529,19114530,19114557,19114562,19109713,19109719,19109723,19109724,19109726,19109731,19190987,19190988,19190989,19190990,19190992]
    data_size = [10,20,50,100,1,5,10,20,50,100,1,5,10,20,50,100,5,10,20,50,100]

    checkpoint_path_ls = []
    out_file_ls = []

    for lm, slurm_id, d_size in zip(ls_lm, training_id, data_size):
        list_of_files = glob.glob(f"/scratch/rv2138/nlu/checkpoints/{slurm_id}/checkpoint*")
        latest_checkpoint_Directory = max(list_of_files, key=os.path.getctime) + '/'
        output_file = lm + str(d_size) + '.out'

        checkpoint_path_ls.append(latest_checkpoint_Directory)
        out_file_ls.append(output_file)

        print(f"Language Model: {lm}")
        print(f"Training Slurm id: {training_id}")
        print(f"Checkpoint Location: {latest_checkpoint_Directory}")
        os.environ["CHECKPOINT_NLU"] = latest_checkpoint_Directory
        os.environ["LM_NLU"] = lm
        os.environ["OUTFILE_NAME_NLU"] = output_file
        os.environ['PATH'] = ':'.join(('/usr/local/bin', os.environ['PATH']))

        # code = f"spark-submit --driver-memory=8g \
        #         --executor-memory=8g \
        #         --executor-cores=40 \
        #         final_tune_parallel.py --rank {param_dic['rank']} --reg {param_dic['regParam']} --alpha {param_dic['alpha']}"
        # code = "sbatch batch_files/run_crows_all.sbatch"

        # os.system(code)
    print(checkpoint_path_ls)
    print(out_file_ls)
    print(ls_lm)
    


# Only enter this block if we're in main
if __name__ == "__main__":
    
    main()
