from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import DistilBertModel, DistilBertTokenizerFast
import torch
import tensorboard
import torch.nn.functional as F
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
import argparse

##################
# READ ARGUMENTS #
##################

parser = argparse.ArgumentParser(description="Set training parameters.")

# Add arguments with default values
parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size for training (default: 128).")
parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation (default: 128).")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs (default: 3).")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer (default: 0.01).")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4).")

# Parse the arguments
args = parser.parse_args()

# Access the values
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
num_train_epochs = args.num_train_epochs
weight_decay = args.weight_decay
learning_rate = args.learning_rate

# Print the parameters (for debugging or confirmation)
print(f"Train Batch Size: {train_batch_size}")
print(f"Eval Batch Size: {eval_batch_size}")
print(f"Number of Epochs: {num_train_epochs}")
print(f"Weight Decay: {weight_decay}")
print(f"Learning Rate: {learning_rate}")



############
# SETTINGS #
############

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#############
# LOAD DATA #
#############

train_dataset = load_from_disk('DATA/ds_train')
valid_dataset = load_from_disk('DATA/ds_valid')

print(train_dataset)
print(valid_dataset)

################
# LOAD OBJECTS #
################

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)

############
# TRAINING #
############

training_args = TrainingArguments(
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    
    evaluation_strategy="epoch",  # Evaluate every few steps
    logging_dir="./logs",         # Directory to save TensorBoard logs
    logging_steps=50,             # Log metrics every 100 steps

    output_dir="./results",        # Directory to save model checkpoints
    save_strategy="epoch",        # Save model checkpoints at the end of each epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer
)

trainer.train()

########
# EVAL #
########

model.eval()

test_dataset = load_from_disk('DATA/ds_test')

scored_dataset = test_dataset.map(lambda examples: model(input_ids=examples['input_ids'].to("cuda:0"), attention_mask=examples['attention_mask'].to("cuda:0")), batched=True, batch_size=eval_batch_size)
test_ce = F.cross_entropy(scored_dataset['logits'], scored_dataset['labels'], reduction='mean')
print(test_ce)

##############
# SAVE MODEL #
##############

model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Set repository name (ensure it's unique)
repo_name = "BERT-sentiment"

# Create a repository on Hugging Face Hub
create_repo(repo_name, exist_ok=True)

# Upload model and tokenizer to the repository
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model uploaded to: https://huggingface.co/{repo_name}")




