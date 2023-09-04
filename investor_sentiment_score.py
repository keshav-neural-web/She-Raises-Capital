# Preprocess text data
import re
import string
import pandas as pd
# BERT embeddings
from transformers import BertModel, BertTokenizer
# Train regression model 
import torch.nn as nn

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\d+', '', text) # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    return text

founders = pd.read_csv("data/founder_v2.csv")
founders = founders.dropna()
founders['cleaned_text'] = founders['text_data'].apply(clean_text)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

founder_vectors = []
for text in founders['cleaned_text']:
   inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
   outputs = model(**inputs)
   founder_vectors.append(outputs.pooler_output[0])
   
founder_vectors = torch.stack(founder_vectors)


class SentimentModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(input_size=768, hidden_size=64, num_layers=1) 
    self.linear = nn.Linear(64,1)
  
  def forward(self, text_vecs):
    lstm_out, _ = self.lstm(text_vecs)
    sent_scores = self.linear(lstm_out[:,-1,:])
    return sent_scores

model = SentimentModel()
model.train() # training loop 
model.eval() # prediction