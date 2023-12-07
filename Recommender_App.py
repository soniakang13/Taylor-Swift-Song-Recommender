#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df_lyrics = pd.read_csv("/Users/soniakang/Desktop/Taylor Swift Song Recommender/songs.csv")
#Clean out Holiday Album
df_lyrics = df_lyrics[df_lyrics['Album'] != 'The Taylor Swift Holiday Collection - EP']
df_lyrics = df_lyrics[df_lyrics['Album']!= 'Cats: Highlights From the Motion Picture Soundtrack']


# In[5]:


#Sentiment Analysis (positive, negative, neutral)
from textblob import TextBlob

#sentiment of lyrics
def calculate_sentiment(lyrics):
    return TextBlob(lyrics).sentiment.polarity

df_lyrics['Sentiment'] = df_lyrics['Lyrics'].apply(calculate_sentiment)


# In[6]:


#Preprocess the lyrics
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize the words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    
    return ' '.join(tokens)


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

#Preprocess all lyrics
df_lyrics['Processed Lyrics'] = df_lyrics['Lyrics'].apply(preprocess_text)

# Prepare data for Doc2Vec
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) 
               for i, _d in enumerate(df_lyrics['Lyrics'])]

# Train a Doc2Vec model
model = Doc2Vec(vector_size=50, min_count=2, epochs=30)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

def recommend_song(input_lyric):
    # Preprocess input lyric
    preprocessed_lyric = preprocess_text(input_lyric)

    # Vector of the input lyric
    input_vector = model.infer_vector(word_tokenize(preprocessed_lyric))

    # Sentiment of the input lyric
    input_sentiment = calculate_sentiment(input_lyric)

    # Find songs with similar vectors
    similar_songs = model.dv.most_similar([input_vector], topn=5)  

    # Refine based on sentiment
    refined_songs = []
    for song_id, similarity in similar_songs:
        song_sentiment = df_lyrics.iloc[int(song_id)]['Sentiment']
        sentiment_diff = abs(song_sentiment - input_sentiment)
        refined_songs.append((song_id, similarity, sentiment_diff))

    # Sort by sentiment difference (lower is better)
    refined_songs.sort(key=lambda x: x[2])

    # Get the most similar song
    most_similar_song_index = int(refined_songs[0][0])
    recommended_song = df_lyrics.iloc[most_similar_song_index]

    return recommended_song['Title']

input_lyric = "I know it's half my fault but I just like to play the victim"
print(recommend_song(input_lyric))


# In[8]:


import streamlit as st


# In[9]:


def display_text_as_html(text, color, size):
    # Using HTML to customize font size and color
    html_string = f"<p style='color: {color}; font-size: {size}px;'>{text}</p>"
    st.markdown(html_string, unsafe_allow_html=True)


# In[12]:


#add background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1500w,f_auto,q_auto:best/rockcms/2023-06/230625-taylor-swift-jm-1236-4fb80d.jpg");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add the background
add_bg_from_url()


# In[13]:


st.title('Taylor Swift Song Recommender')

#User Input
input_lyric = st.text_input("Enter a lyric:")

if st.button('Get Recommendation'):
    if input_lyric:
        # Call the recommend_song function and display the result
        recommended_song = recommend_song(input_lyric)
        display_text_as_html(f"Recommended Song: {recommended_song}", "white", 30)
    else:
        st.write("Please enter a lyric to get a recommendation.")

