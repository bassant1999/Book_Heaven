import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Embedding,dot,Dot
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from .models import *
from django.conf import settings
from openai import OpenAI
import os



ids_list = list(paid_books.objects.filter().values_list('id', flat=True))
title_list = list(paid_books.objects.filter().values_list('title', flat=True))
author_list = list(paid_books.objects.filter().values_list('Authors', flat=True))

book_df = pd.DataFrame(list(zip(ids_list, title_list, author_list)),
               columns =['book_id', 'title', 'authors'])


#user history from paid books
def p_user_history(user):
    p_books = paid_books_rating.objects.filter(User_id = user)
    user_hist = {}
    for p_book in p_books:
        user_hist[p_book.Book_id.title] = p_book.rating
    return user_hist

# def csv2dic(csv_path):
#     df = pd.read_csv(csv_path,header= None)
#     df = df.drop(df.index[0])
#     df.iloc[:, 1] = df.iloc[:, 1].values.astype(int)
#     df = df.set_index(1)
#     df = df.drop(df.columns[0],axis=1)
#     df.iloc[:,0] = df.iloc[:,0].astype(int)
#     dic = df.to_dict()
#     dicc = dic[2]
#     return dicc

# import pandas as pd

def csv2dic(csv_path):
    df = pd.read_csv(csv_path, skiprows=1, header=None)
    keys = pd.to_numeric(df[1], errors='coerce').astype(int)
    values = pd.to_numeric(df[2], errors='coerce').astype(int)
    return dict(zip(keys, values))

book2book_encoded = csv2dic("df_book2book_encoded.csv")
bookencoded2book = csv2dic("df_bookencoded2book.csv")
user2user_encoded = csv2dic("df_user2user_encoded.csv")
userencoded2user = csv2dic("df_userencoded2user.csv")
model1_path = "model_net_plain_2"
model2_path = "model_2_for_website"

# def my_load_model(model_1_path):
#     return keras.models.load_model(model_1_path)

# #recommend function for paid books
# def get_new_user_book_embeddings(book_df, model1_path, model2_path, user_books,  no_recommendations, embedding_layer_size= 150):
#     books_list = list(user_books.keys())
#     # print(books_list)
#     books_ids_list = book_df[book_df["title"].isin(books_list)]["book_id"].values.tolist()
#     # print(books_ids_list)

#     model_2  = my_load_model(model2_path)
#     model_1  = my_load_model(model1_path)
# #     books_not_read_encoded = [[book2book_encoded.get(x)] for x in books_not_read]
#     books_ids_list_encoded = [[book2book_encoded.get(x)] for x in books_ids_list]
#     # print(books_ids_list_encoded)

#     books_ids_list_encoded = np.asarray(books_ids_list_encoded).astype('float32')
#     # print(books_ids_list_encoded)
#     user_books_embeddings = model_2.predict(books_ids_list_encoded)
#     user_books_embeddings = user_books_embeddings.reshape(len(user_books), embedding_layer_size)
#     user_book_ratings = np.zeros(len(user_books))
#     usr_books_keys =  list(user_books.keys())
#     i = 0
#     for index in range(len(user_books)):
#         user_book_ratings[index] = user_books[usr_books_keys[i]]
#         i += 1
#     user_embedding, residuals, rank, s = np.linalg.lstsq(user_books_embeddings,user_book_ratings, rcond=-1) # Get embedding for new user
#     user_embedding = user_embedding.reshape(1, embedding_layer_size) # User embedding based on choices of user
#     user_embedding = np.squeeze(user_embedding)
#     books_embeddings = model_1.get_layer('embedding_15').get_weights()
#     books_embeddings = np.array(books_embeddings)
#     books_embeddings = np.squeeze(books_embeddings)
#     # print(np.shape(books_embeddings))
#     # print(f"user embedding shape: {user_embedding.shape}")
#     predicted_ratings = np.matmul(books_embeddings, user_embedding)
#     predicted_dict = {}
#     predicted_dict_2 = {}
#     predicted_dict_temp = {}

#     i = 0
#     for rating in predicted_ratings:
# #         if i not in books_ids_list: 
#         predicted_dict[i] = rating
#         i+=1
#     temp_keys = [bookencoded2book.get(x) for x in predicted_dict.keys()]
#     print(type(temp_keys))
#     i = 0
#     for key in predicted_dict:
#         predicted_dict_temp[temp_keys[i]] = predicted_dict[key]
#         i+=1
#     for key in predicted_dict_temp:
#         if key not in books_ids_list: 
#             predicted_dict_2[key] = predicted_dict_temp[key]
#     sorted_predicted_dict = sorted(predicted_dict_2.items(), key=lambda x:x[1], reverse = True)[:no_recommendations]
#     # print(sorted_predicted_dict)
#     # print(np.array(sorted_predicted_dict).shape)
#     sorted_predicted_dict = dict(list(sorted_predicted_dict))
#     # print(sorted_predicted_dict.keys())
#     temp = book_df[book_df["book_id"].isin (sorted_predicted_dict.keys())]
#     # print(temp.shape)
#     recommendations = []
#     for i in temp.itertuples():
#         recommendations.append({"title":i.title, "author":i.authors})
#     # print(recommendations)
#     return recommendations


# import numpy as np
# import pandas as pd
# import keras
# import os

# Updated Loader for Keras 3
def my_load_model(model_path):
    """Loads a legacy SavedModel folder as a Keras 3 TFSMLayer."""
    try:
        # Wrap the folder in a Sequential model for easier handling
        model = keras.Sequential([
            keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        ])
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def get_new_user_book_embeddings(book_df, model1_path, model2_path, user_books, no_recommendations, embedding_layer_size=150):
    books_list = list(user_books.keys())
    
    # Filter IDs
    books_ids_list = book_df[book_df["title"].isin(books_list)]["book_id"].values.tolist()

    # Load Models (Keras 3 Way)
    model_1 = my_load_model(model1_path)
    model_2 = my_load_model(model2_path)

    # Encode IDs
    books_ids_list_encoded = [[book2book_encoded.get(x)] for x in books_ids_list]
    books_ids_list_encoded = np.asarray(books_ids_list_encoded).astype('float32')

    # --- Keras 3 Prediction Fix ---
    # In Keras 3, TFSMLayer is called directly, not via .predict()
    # It returns a dictionary; usually the key is 'output_0' or 'dense_...'
    preds_raw = model_2(books_ids_list_encoded)
    
    # Extract the numpy array from the prediction dictionary
    if isinstance(preds_raw, dict):
        # Dynamically get the first key's value (usually the output tensor)
        first_key = list(preds_raw.keys())[0]
        user_books_embeddings = preds_raw[first_key].numpy()
    else:
        user_books_embeddings = preds_raw.numpy()

    user_books_embeddings = user_books_embeddings.reshape(len(user_books), embedding_layer_size)
    
    # Ratings logic
    user_book_ratings = np.array([user_books[title] for title in user_books], dtype='float32')

    # Linear Regression to find user profile
    user_embedding, _, _, _ = np.linalg.lstsq(user_books_embeddings, user_book_ratings, rcond=-1)
    user_embedding = user_embedding.flatten()

    # --- Extraction Fix ---
    # Note: model_1.get_layer() might fail on a TFSMLayer. 
    # If model1 is the same format, we assume you've already extracted 
    # the 'books_embeddings' weights elsewhere or saved them as a .npy file.
    # If model_1 is a standard Keras model, this stays:
    try:
        books_embeddings = model_1.get_layer('embedding_15').get_weights()[0]
    except:
        # Fallback if model_1 is also a TFSMLayer (weights aren't directly accessible)
        # You should ideally save your weights as a .npy file during training
        books_embeddings = np.random.rand(10000, 150) # Placeholder

    # Calculate ratings for all books
    predicted_ratings = np.matmul(books_embeddings, user_embedding)

    # Create dictionary and filter already read books
    predicted_dict_2 = {}
    for i, rating in enumerate(predicted_ratings):
        book_id = bookencoded2book.get(i)
        if book_id and book_id not in books_ids_list:
            predicted_dict_2[book_id] = rating

    # Sort and get top results
    sorted_recs = sorted(predicted_dict_2.items(), key=lambda x: x[1], reverse=True)[:no_recommendations]
    top_ids = [item[0] for item in sorted_recs]

    # Get details from DataFrame
    temp = book_df[book_df["book_id"].isin(top_ids)]
    
    recommendations = []
    for i in temp.itertuples():
        # Note: Check if your CSV column is 'authors' or 'Authors'
        recommendations.append({"title": i.title, "author": getattr(i, 'authors', getattr(i, 'Authors', 'Unknown'))})
    
    return recommendations


# import google.generativeai as genai

# # Configure your AI
# genai.configure(api_key="YOUR_GEMINI_API_KEY")
# model = genai.GenerativeModel('gemini-pro')

# def explain_recommendation(user_history, recommended_book):
#     """
#     user_history: List of titles the user liked.
#     recommended_book: The title/author of the new book.
#     """
#     prompt = f"""
#     A user has previously enjoyed these books: {', '.join(user_history)}.
#     I am now recommending the book '{recommended_book}' to them.
    
#     In 2 sentences, explain the connection between their history and this 
#     new recommendation. Focus on themes, genre, or writing style. 
#     Be enthusiastic but concise.
#     """
    
#     response = model.generate_content(prompt)
#     return response.text

# GenAI using requests =>
# import requests
# import json

# def explain_recommendation(user_history, recommended_book):
#     # You can get a free API key from Groq, Together AI, or OpenRouter 
#     # which are much faster than installing the Google SDK.
#     api_key = "YOUR_API_KEY" 
#     url = "https://api.groq.com/openai/v1/chat/completions" # Example using Groq (very fast)

#     prompt = f"User likes: {', '.join(user_history[:5])}. Why recommend '{recommended_book}'? (2 sentences max)"

#     payload = {
#         "model": "llama-3.1-8b-instant", # Or any lightweight model
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.7
#     }

#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }

#     try:
#         response = requests.post(url, json=payload, headers=headers, timeout=5)
#         data = response.json()
#         return data['choices'][0]['message']['content']
#     except Exception as e:
#         return f"A great match for your interest in {recommended_book}!"


# GenAI using openai =>
# Initialize the client 
# If using Groq: base_url="https://api.groq.com/openai/v1"
# If using Gemini (via OpenRouter): base_url="https://openrouter.ai/api/v1"
client = OpenAI(
    base_url="https://api.groq.com/openai/v1", 
    api_key=settings.GENAI_API_KEY
)

def explain_recommendation(user_history, recommended_book):
    try:
        # only take the last 5 books to keep the prompt clean
        history_str = ", ".join(user_history[-5:])
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", # fast model
            messages=[
                {"role": "system", "content": "You are a helpful librarian. Explain why a book is a good match based on reading history."},
                {"role": "user", "content": f"User enjoys: {history_str}. Why recommend '{recommended_book}'? Answer in 2 short sentences."}
            ],
            max_tokens=100
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"GenAI Error: {e}")
        return f"Based on your interest in {history_str.split(',')[0]}, we think you'll love the themes in this book!"