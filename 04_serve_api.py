from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Initialize FastAPI app
app = FastAPI()

################
# LOAD OBJECTS #
################

tokenizer = AutoTokenizer.from_pretrained("TomekTarczynski/BERT-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("TomekTarczynski/BERT-sentiment")

# Define id2label and label2id mappings
id2label = {
    "0": "negative",
    "1": "neutral",
    "2": "positive"
}
label2id = {v: k for k, v in id2label.items()}  # Reverse mapping

# Assign mappings to the model's configuration
model.config.id2label = id2label
model.config.label2id = label2id

##################
# DEFINE SERVING #
##################

def calculate_sentiment(text: str):
    """
    Predict sentiment for the given text using the loaded model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class = model.config.id2label[str(torch.argmax(probs, dim=-1).item())]
    formatted_probs = [f"{p:.4f}" for p in probs.squeeze().detach().numpy()]
    return predicted_class, formatted_probs

##################
# API ENDPOINTS  #
##################

# Define input model for validation
class SentimentRequest(BaseModel):
    text: str

# Define response model
class SentimentResponse(BaseModel):
    sentiment: str
    probabilities: list

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for the input text.
    """
    sentiment, probabilities = calculate_sentiment(request.text)
    return SentimentResponse(sentiment=sentiment, probabilities=probabilities)
