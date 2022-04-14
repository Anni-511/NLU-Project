from datasets import *
import os
import json
import nltk
from tqdm import tqdm

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
        'him' : 'her',
        'her' : 'him',
        'his' : 'hers',
        'hers' : 'his',
        'he' : 'she',
        'she' : 'he',
        'himself' : 'herself',
        'herself' : 'himself'
    }
    # dataset = load_dataset('wikitext', 'wikitext-103-v1', split = 'train')
    # dataset_to_swapped_text(dataset, dic_swap= dic_swap, output_filename="swapped_data.txt")
    example_dataset = {'text': ['she is testing',
                            'He is eating', 
                            'she is talking with him', 
                            'is she, talking to him?', 
                            'She told her, she should go to tge concert', 
                            'is she talking to herself',
                            "she's talking to him"]}

    dataset_to_swapped_text(example_dataset, dic_swap = dic_swap, output_filename= "train_try.txt")

    print("Success")