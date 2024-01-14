import pandas as pd
import nltk
import re
import numpy as np
from transformers import pipeline
import torch
 
emotionModel = torch.load("./emotion.pt")
 
nltk.download('punkt')
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
 
def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc
 
def get_top_emotions(text, n=2):
    emotions = emotionModel(text)
    if emotions and emotions[0]:
        # Sorting emotions based on scores in descending order
        sorted_emotions = sorted(emotions[0], key=lambda x: x['score'], reverse=True)
        # Extracting the top n emotions and their scores
        top_emotions = [(emotion.get('label'), emotion.get('score')) for emotion in sorted_emotions[:n]]
        return top_emotions
   
 
def infer_emotions(given_text):
   
 
    primary_emotion = []
    secondary_emotion = []
    for k in given_text:
        given_text = normalize_document(k)
        emothon_output = get_top_emotions(k, n=2)
        primary_emotion.append(emothon_output[0][0])
        secondary_emotion.append(emothon_output[1][0])
   
 
    a = list(set(primary_emotion))
    b = list(set(secondary_emotion))
    check = ['anger','fear', 'joy', 'neutral', 'sadness', 'surprise',]
 
    final_primary_list =[]
    for i in check:
        if i in a:
            final_primary_list.append(1)
        else:
            final_primary_list.append(0)
 
    final_secondary_list =[]
    for i in check:
        if i in b:
            final_secondary_list.append(1)
        else:
            final_secondary_list.append(0)
    return final_primary_list + final_secondary_list