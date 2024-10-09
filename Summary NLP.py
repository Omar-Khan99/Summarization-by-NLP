from modeling_deltalm import DeltalmForConditionalGeneration  
from configuration_deltalm import DeltalmConfig      
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional,SimpleRNN
import nltk
import json
from rouge_score import rouge_scorer


model = DeltalmForConditionalGeneration.from_pretrained("hhhhzy/deltalm-base-xlsum")
tokenizer = AutoTokenizer.from_pretrained("hhhhzy/deltalm-base-xlsum")


paragraph= []
summary = []
# Open a file in read mode

with open("dataset\\labeled_validation_dataset.jsonl", "r", encoding="utf-8") as f:
    # Iterate over the lines in the file
    for line in f:
        # Parse each line as a JSON object
        obj = json.loads(line)
        # Do something with the object
        paragraph.append(obj["paragraph"])
        summary.append(obj["summary"])
 

def len_text(text):
    length = len(text)
    min_ = 0.10 * length 
    max_ = 0.17 * length 
    min_att = 2**round(np.log2(min_))
    max_att = 2**round(np.log2(max_))
    return min_att,max_att


min_0 , max_0 = len_text(paragraph[0])
inputs = tokenizer(paragraph[0], max_length=max_0, return_tensors="pt",truncation=True)

generate_ids = model.generate(inputs["input_ids"], min_length=min_0, max_length=max_0)
t_0=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)[0]


# Instantiate the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(summary[0], t_0)

# Print ROUGE scores
print(scores['rouge1'])
print(scores['rouge2'])
print(scores['rougeL'])