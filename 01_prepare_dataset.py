from datasets import load_dataset
from collections import Counter
from transformers import DistilBertModel, DistilBertTokenizerFast
import torch
assert torch.cuda.is_available() == True

############
# SETTINGS #
############

min_chars = 5 # Minimum number of chars in text to check sentiment
max_chars = 400 # Maximum number of chars in text to check sentiment
min_char_count = 100 # If character occurs lower number of times then it is removed
max_tokens = 128 # Maximum number of tokens after tokenizing text
num_proc = 7 # Number of processes during map

################
# LOAD OBJECTS #
################

# Amazon-reviews dataset will be used to finetune DistilBert model to sentiment analisys
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)['full']
# distilbert-base-uncased will be finetune
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

#################
# PREPROCESSING #
#################

dataset = dataset.map(lambda row: {**row, "labels": 2 if row["rating"] >= 4 else 1 if row["rating"] == 3 else 0}, num_proc=num_proc)
dataset = dataset.map(lambda row: {"text": row["text"].lower()}, num_proc=num_proc)
dataset = dataset.remove_columns(['title', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase'])

# Remove rare characters
cnt = Counter(''.join(dataset['text']))
chars_to_remove = []
for char, count in sorted(cnt.items()):
    if count < 100:
        chars_to_remove.append(char)
dict_trans = str.maketrans('', '', ''.join(chars_to_remove))
dataset = dataset.map(lambda row, dict_trans: {"text": row["text"].translate(dict_trans)}, fn_kwargs={"dict_trans": dict_trans}, num_proc=num_proc)

dataset = dataset.map(lambda row: {"len_text": len(row['text'])}, num_proc=num_proc)
dataset = dataset.filter(lambda row: (row['len_text'] < max_chars) & (row['len_text'] >= min_chars), num_proc=num_proc)
dataset = dataset.map(lambda row, tokenizer: tokenizer(row['text'], padding=True, truncation=True), batched=False, fn_kwargs={"tokenizer": tokenizer}, num_proc=num_proc)
dataset = dataset.map(lambda row: {'n_tokens': len(row['input_ids'])}, num_proc=num_proc)
dataset = dataset.filter(lambda row: row['n_tokens'] <= max_tokens, num_proc=num_proc)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

##################
# SPLIT AND SAVE #
##################

train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
train_dataset = train_test_split['train']
test_dataset = test_valid_split['test']
valid_dataset = test_valid_split['train']

train_dataset.save_to_disk('DATA/ds_train')
valid_dataset.save_to_disk('DATA/ds_valid')
test_dataset.save_to_disk('DATA/ds_test')