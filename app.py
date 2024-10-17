#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py
import streamlit as st
import joblib


# In[2]:


# Load the pre-trained model and vectorizer
model = joblib.load( 'sentiment_model.pkl')  # Ensure this file is in the same directory or provide the full path
vectorizer = joblib.load( 'tfidf_vectorizer.pkl')


# In[3]:


# Streamlit UI
st.title("Sentiment Analysis Application")

# User input
user_input = st.text_area("Enter a review for sentiment analysis:")


# In[ ]:


# Add a button for prediction
if st.button("Predict"):
    if user_input:  # Check if the user has entered any input
        # Transform the input using the loaded vectorizer
        transformed_input = vectorizer.transform([user_input])
        
        # Predict the sentiment
        prediction = model.predict(transformed_input)[0]
        
        # Present the result more nicely
        st.subheader("Prediction Result:")
        
        # Display different sentiments with emojis
        if prediction == 2:
            st.write("**Sentiment:** Positive 😄")
        elif prediction == 0:
            st.write("**Sentiment:** Negative 😞")
        else:
            st.write("**Sentiment:** Neutral 😐")

        # Display the input review
        st.write("**Your Review:**")
        st.write(f"\"{user_input}\"")

    else:
        st.warning("Please enter a review before clicking the Predict button.")

