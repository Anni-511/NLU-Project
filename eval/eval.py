import os
import sys
import json
import argparse
import time

sys.path.insert(1, './jiant')

import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default= './config/run_mrpc.json')
	parser.add_argument('--model-path', type=str, default= None)
	args = parser.parse_args()
	f = open(args.config)
	config = json.load(f)
	os.makedirs(config['data_dir'], exist_ok=True)
	os.makedirs(config['exp_dir'], exist_ok=True)

	tasks = ['cb', 'copa', 'rte', 'wic', 'wsc', 'boolq', 
			'superglue_broadcoverage_diagnostics', 
			'superglue_winogender_diagnostics']
	# tasks = ['rte']
	# multirc, cb, copa, record, rte, wic, wsc, boolq, superglue_broadcoverage_diagnostics, superglue_winogender_diagnostics
	results = []
	if args.model_path:
		config['model_weights_path'] = args.model_path
	for task in tasks:
		print(f"--------Running for {task}--------")
		start_time = time.time()
		config["tasks"] =  task
		config['train_tasks'] = [task]
		config['val_tasks'] = [task]
		config['test_tasks'] = [task]
		config["run_name"] =  f"{task}_roberta-base"
		downloader.download_data(config['train_tasks'], config['data_dir'])

		new_config = f'./config/run_{task}.json'
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
	results_file_name = config['hf_pretrained_model_name_or_path']
	results_file = f'./{results_file_name}.json'
	with open(results_file, 'w') as f:
		json.dump(results, f, indent= 4)
	print("Success")