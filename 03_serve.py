from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import argparse

##################
# READ ARGUMENTS #
##################

parser = argparse.ArgumentParser(description="Set training parameters.")
parser.add_argument("--text", type=str, default="default_text", help="Text input (default: 'default_text').")
args = parser.parse_args()
text = args.text

################
# LOAD OBJECTS #
################

tokenizer = AutoTokenizer.from_pretrained("TomekTarczynski/BERT-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("TomekTarczynski/BERT-sentiment")

# The mapping has not been configured during training
id2label = {
    "0": "negative",
    "1": "neutral",
    "2": "positive"
  }
label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
  }

model.config.id2label = id2label
model.config.label2id = label2id

##################
# DEFINE SERVING #
##################

def calculate_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class = model.config.id2label[str(torch.argmax(probs, dim=-1).item())]
    return predicted_class, probs

print(calculate_sentiment(text))
