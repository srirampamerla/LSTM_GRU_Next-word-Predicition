# to deploy in the stram lit app
## Write in Prediction .ipynb
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the LSTM MOdel
model=load_model('next_word_lstm.h5')

# Load the Tokenizer
with open('tokenizer.pickle','rb')as handle:
  tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_seq_len):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if len(token_list)>=max_seq_len:
    token_list=token_list[-(max_seq_len-1):]
  token_list=pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(predicted,axis=1)
  for word,index in tokenizer.word_index.items():
    if index==predicted_word_index:
      return word
  return None

# STream lit app

st.title('Next Word Prediction with LSTM and Early Stopping')
input_text=st.text_input("Enter the sequence of words","To be or not")
if st.button('Predict Next Word'):
  maxi_seq_len=model.input_shape[1]+1 # Reterive the max_seq_length from the model input shape
  next_word=predict_next_word(model,tokenizer,input_text,maxi_seq_len)
  st.write(f'Next Word : {next_word}')
