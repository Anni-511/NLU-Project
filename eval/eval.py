# 19114044 d 19114331 d 19114448 d 19114459 d 19114460 d 19114476 d BERT
# 19109713 19109719 19109723 19109724 19109726 19109731 Roberta
# 19190985 d 19190987 d 19190988 d 19190989 19223411 Deberta v3
# 19298407 19298413 19298445 19298449 19298455 19298464 19298505 19298510 19298534 BERT PROP
# 19202307 19202306 19202304 19202299 19202283 19202282 19202277 19202280 19202274 Deberta v3 PROP
import os
import sys
import json
import argparse
import time
import torch
import pandas as pd

sys.path.insert(1, './jiant')

import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader

model_dict = {'BERT': 'bert-base-uncased',
			  'DistilBERT': 'distilbert-base-uncased',
			  'RoBERTa': 'roberta-base',
			  'DeBERTa': 'microsoft/deberta-base',
			  'DeBERTa-v3': 'microsoft/deberta-v3-base'}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default= './config/run_mrpc.json')
	parser.add_argument('--slurm-id', type=str, default= None)
	parser.add_argument('--model', type=str, default= None)
	args = parser.parse_args()
	f = open(args.config)
	config = json.load(f)
	os.makedirs(config['data_dir'], exist_ok=True)
	os.makedirs(config['exp_dir'], exist_ok=True)

	print(args.slurm_id)
	df = pd.read_csv('./runs_prop.csv')

	tasks = ['cb', 'copa', 'rte', 'wic', 'wsc', 'boolq','multirc']
			# 'superglue_broadcoverage_diagnostics', 
			# 'superglue_winogender_diagnostics']
	# tasks = ['boolq']
	# multirc, cb, copa, record, rte, wic, wsc, boolq, superglue_broadcoverage_diagnostics, superglue_winogender_diagnostics
	results = []
	
	if not args.slurm_id and not args.model:
		sys.exit("Both slurm id and model not provided. Provide atleast one.")

	if args.slurm_id:
		# Get model path
		model_root = f'/scratch/rv2138/nlu/checkpoints/{args.slurm_id}/'
		model_path = f'/scratch/rv2138/nlu/checkpoints/{args.slurm_id}/' + max(next(os.walk(model_root))[1]) + '/pytorch_model.bin'
		m = torch.load(model_path, map_location = 'cpu')
		new_model_path = model_path[:-4] + '_converted.pt'
		torch.save(m, new_model_path)
		config['model_weights_path'] = new_model_path
		model_name = df[df['Training Experiment ID']==int(args.slurm_id)].iloc[0]['Model']
	else:
		args.slurm_id = 11111111
		model_name = args.model
		
	model_name = model_dict[model_name]

	for task in tasks:
		m_name = model_name.replace("/", "")
		print(f"--------Running for {task}--------")
		start_time = time.time()
		config["tasks"] =  task
		config['train_tasks'] = [task]
		config['val_tasks'] = [task]
		config['test_tasks'] = [task]
		config['hf_pretrained_model_name_or_path'] = model_name
		config["run_name"] =  f"{task}_{m_name}_{args.slurm_id}"
		downloader.download_data(config['train_tasks'], config['data_dir'])

		new_config = f'./config/run_{task}_{m_name}_{args.slurm_id}.json'
		with open(new_config, 'w') as f:
			json.dump(config, f, indent= 4)
		args_eval = simple_run.RunConfiguration.from_json_path(new_config)
		simple_run.run_simple(args_eval)

		run_name = config['run_name']
		results_file = f'/scratch/rv2138/nlu/superglue/exp/runs/{run_name}/val_metrics.json'
		with open(results_file, 'r') as f:
			results.append(json.load(f))
		results[-1]['time_taken'] = (time.time() - start_time)/60
	print("Writing results file")
	# results_file_name = config['hf_pretrained_model_name_or_path']
	results_file = f'/scratch/rv2138/nlu/superglue/results_10/{m_name}_{args.slurm_id}.json'
	with open(results_file, 'w') as f:
		json.dump(results, f, indent= 4)
	print("Success")