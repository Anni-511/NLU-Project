from datasets import *
import os
import json
import nltk
from tqdm import tqdm
import numpy as np

def dataset_to_text(dataset, output_filename="data.txt"):
	"""Utility function to save dataset text to disk,
	useful for using the texts to train the tokenizer 
	(as the tokenizer accepts files)"""
	with open(output_filename, "w") as f:
		for t in dataset["text"]:
			print(t, file=f)

def swap_gender_sentence(sentence, dic_swap= None):
	nltk.download('punkt')
	sentence_split = nltk.word_tokenize(sentence)
	new_sentence = []
	for word in sentence_split:
		if word.lower() in dic_swap.keys():
			new_sentence.append(dic_swap[word.lower()])
		else:
			new_sentence.append(word)
	return " ".join(new_sentence)

def swap_prop_dataset(dataset, proportion=0.1, dic_swap=None, output_filename=None):
	idx = np.arange(0, len(dataset), 1)
	num_points_to_swap = int(proportion * len(dataset))
	swap_idx = set(np.random.choice(idx, size=num_points_to_swap, replace = False))
	with open(output_filename, "w") as f:
		for i, t in enumerate(tqdm(dataset["text"])):
			if i in swap_idx:
				new_sentence = swap_gender_sentence(t, dic_swap)
			else:
				new_sentence = t
			print(new_sentence, file=f)

def dataset_to_swapped_text(dataset, dic_swap= None, output_filename= None):
	"""Utility function to save dataset text to disk,
	useful for using the texts to train the tokenizer 
	(as the tokenizer accepts files)"""
	with open(output_filename, "w") as f:
		for t in tqdm(dataset["text"]):
			if dic_swap:
				swapped_sentence = swap_gender_sentence(t,dic_swap)
			else:
				swapped_sentence = t
			print(swapped_sentence, file=f)

if __name__ == '__main__':
	"""
	dict_path: 
	"""
	print("Started")
	dic_swap = {
		'actor': 'actress',
		'actress': 'actor',
		'boy': 'girl',
		'girl': 'boy',
		'boyfriend': 'girlfriend',
		'girlfriend': 'boyfriend',
		'boys': 'girls',
		'girls': 'boys',
		'father': 'mother',
		'mother': 'father',
		'fathers': 'mothers',
		'mothers': 'fathers',
		'gentleman': 'lady',
		'lady': 'gentleman',
		'gentlemen': 'ladies',
		'ladies': 'gentlemen',
		'grandson': 'granddaughter',
		'granddaughter': 'grandson',
		'he': 'she',
		'she': 'he',
		'hero': 'heroine',
		'heroine': 'hero',
		'him': 'her',
		'her': 'his',
		'his': 'her',
		'husband': 'wife',
		'wife': 'husband',
		'husbands': 'wives',
		'wives': 'husbands',
		'king': 'queen',
		'queen': 'king',
		'kings': 'queens',
		'queens': 'kings',
		'male': 'female',
		'female': 'male',
		'males': 'females',
		'females': 'males',
		'man': 'woman',
		'woman': 'man',
		'men': 'women',
		'women': 'men',
		'mr.': 'mrs.',
		'mrs.': 'mr.',
		'prince': 'princess',
		'princess': 'prince',
		'son': 'daughter',
		'daughter': 'son',
		'sons': 'daughters',
		'daughters': 'sons',
		'spokesman': 'spokeswoman',
		'spokeswoman': 'spokesman',
		'stepfather': 'stepmother',
		'stepmother': 'stepfather',
		'uncle': 'aunt',
		'aunt': 'uncle',
		'himself' : 'herself',
		'herself' : 'himself'
	}
	dataset = load_dataset('wikitext', 'wikitext-2-v1', split = 'train')
	train_size = 0.99999
	dataset = dataset.train_test_split(test_size= 1 - train_size, seed = 42)['train']
	# dataset_to_swapped_text(dataset,
	# 						dic_swap= dic_swap, 
	# 						output_filename="wiki_2_swapped_wiki_50.txt")
	prop_swap = 0.95
	swap_prop_dataset(dataset, proportion=prop_swap, dic_swap=dic_swap, output_filename=f'wiki_2_prop_{int(prop_swap*100)}_{int(100 - prop_swap*100)}.txt')
	# example_dataset = {'text': ['she is testing',
	#                         'He is eating', 
	#                         'she is talking with him', 
	#                         'is she, talking to him?', 
	#                         'She told her, she should go to tge concert', 
	#                         'is she talking to herself',
	#                         "she's talking to him"]}

	# dataset_to_swapped_text(example_dataset, dic_swap = dic_swap, output_filename= "train_try.txt")

	print("Success")
