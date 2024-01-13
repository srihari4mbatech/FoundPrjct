import streamlit as st
import numpy as np
import pandas as pd
import spacy
import joblib
import time
# from spacy_download import load_spacy
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
 
# Load the pre-trained model and scaler
loaded_model = load_model("stock_price_prediction_model.h5")
loaded_scaler = joblib.load("scaler.joblib")
scaler = MinMaxScaler()  # Assuming you have saved the scaler during training
# nlp= load_spacy('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

import requests

API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
headers = {"Authorization": "Bearer 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"}

EMOT_API_URL= "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"



def query(URL,payload):
	response = requests.post(URL, headers=headers, json=payload)
	return response.json()
	
# output = query({
# 	"inputs": "I like you. I love you",s
# })

import pandas as pd
import nltk
import re
import numpy as np
from transformers import pipeline
import torch
 
# emotionModel = torch.load("emotion.pt")
 
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
    # emotions = emotionModel(text)
    emotions=query(EMOT_API_URL,{"text":text})
    time.sleep(1)
    # st.text(emotions)
    if emotions:
        # Sorting emotions based on scores in descending order
        # st.text(emotions)
        sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
        # Extracting the top n emotions and their scores
        top_emotions = [(emotion.get('label'), emotion.get('score')) for emotion in sorted_emotions[:n]]
        # st.text(top_emotions)
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
# [00:26] Thrinath Nelaturi
# given_text = ["he is a good person","he is a bad person","they are good ","they are bad"]
# infer_emotions(given_text)



####


 
def preprocess_input_data(open_val, high_val, low_val, volume_val, close_val,
                          headline, content):
    # Assume you have a function to process the news data and extract sentiment scores
    positive_sentiment_val, negative_sentiment_val, neutral_sentiment_val = process_news_data(headline, content)
    emotion_lst=infer_emotions([headline])
 
    # Reshape input data
    input= [open_val, high_val, low_val, volume_val, close_val, positive_sentiment_val, negative_sentiment_val, neutral_sentiment_val]+emotion_lst
    input_data = np.array([input])
 
    # Scale the input data
    scaled_input_data = loaded_scaler.transform(input_data)
 
    return scaled_input_data
 
def process_news_data(headline, content):
    # Placeholder function, replace with actual sentiment analysis logic
    #def genSentforpsg(StrText,WordLst):
    # WordLst= ['Exxon Mobil','Exxon']
    outputSentLst=[]
    docs=nlp(headline+" "+content)
    for sent in docs.sents:
        # for ent in sent.ents:
        #     if ent.text in WordLst:
        output=query(API_URL,{"inputs":sent.text[0:512],})
        time.sleep(2)
        # st.text(output[0])
        outputSentLst=outputSentLst + output[0]
    sentDf= pd.DataFrame(outputSentLst)
    if 'label' in sentDf.columns: 
        sentDict=sentDf.groupby('label')['score'].mean().to_dict()
    else:
        sentDict= {'positive':0,}
    #return sentDict
    positive_sentiment_val,negative_sentiment_val,neutral_sentiment_val = 0,0,0
    if 'positive' in sentDict.keys():
         positive_sentiment_val = sentDict['positive']
    if 'negative' in sentDict.keys():
        negative_sentiment_val = sentDict['negative']
    if 'neutral' in sentDict.keys():
        neutral_sentiment_val = sentDict['neutral']
    # st.text(str(sentDict['positive'])+" "+str(sentDict['negative'])+" "+str(sentDict['neutral']))
    return positive_sentiment_val, negative_sentiment_val, neutral_sentiment_val
 
def main():
    st.title("Stock Price Prediction App")
 
    # Input form for user input
    open_val = st.number_input("Enter Open Value:")
    high_val = st.number_input("Enter High Value:")
    low_val = st.number_input("Enter Low Value:")
    volume_val = st.number_input("Enter Volume Value:")
    close_val = st.number_input("Enter Close Value:")
 
    headline = st.text_input("Enter News Article Headline:")
    content = st.text_area("Enter News Article Content:")
 
    # Button to generate prediction
    if st.button("Generate Prediction"):
        # Preprocess input data
        input_data = preprocess_input_data(open_val, high_val, low_val, volume_val, close_val,
                                           headline, content)
 
        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)
 
        # Display the prediction
        st.success(f"Predicted Next_Week_Close: {prediction.flatten()[0]}")
 
if __name__ == "__main__":
    main()