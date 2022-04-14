from datasets import *
from transformers import tokenizer
import os
import json

def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["text"], return_special_tokens_mask=True)

if __name__ == '__main__':
    files = ["train_try.txt"]
    dataset = load_dataset("text", data_files=files, split="train")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    print("Success")