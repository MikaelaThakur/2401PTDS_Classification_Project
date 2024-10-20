import pandas as pd
import numpy as np
import time
import streamlit as st
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Load the datasets
st.write('Loading datasets...')
train_articles = pd.read_csv('https://raw.githubusercontent.com/Jana-Liebenberg/2401PTDS_Classification_Project/refs/heads/main/Data/processed/train.csv')  
test_articles = pd.read_csv('https://raw.githubusercontent.com/Jana-Liebenberg/2401PTDS_Classification_Project/refs/heads/main/Data/processed/test.csv')    

# Drop the 'url' column
st.write('Dropping URL column...')
train_articles.drop(columns=['url'], inplace=True, errors='ignore')
test_articles.drop(columns=['url'], inplace=True, errors='ignore')

# Combine text columns for training and test sets
train_articles['combined_text'] = train_articles['headlines'] + ' ' + train_articles['description'] + ' ' + train_articles['content']
test_articles['combined_text'] = test_articles['headlines'] + ' ' + test_articles['description'] + ' ' + test_articles['content']

# Convert all text in the 'combined_text' column to lowercase
st.write('Lowering case...')
train_articles['combined_text'] = train_articles['combined_text'].str.lower()
test_articles['combined_text'] = test_articles['combined_text'].str.lower()

# Define a function to remove punctuation and numbers
st.write('Cleaning punctuation...')
def remove_punctuation_numbers(post):
    punc_numbers = string.punctuation + '0123456789'
    return ''.join([l for l in post if l not in punc_numbers])

train_articles['combined_text'] = train_articles['combined_text'].apply(remove_punctuation_numbers)
test_articles['combined_text'] = test_articles['combined_text'].apply(remove_punctuation_numbers)

# Vectorize the text
vect = CountVectorizer(stop_words='english')
X_train = vect.fit_transform(train_articles['combined_text'])
X_test = vect.transform(test_articles['combined_text'])

# Encode the target variable
le = LabelEncoder()
y_train = le.fit_transform(train_articles['category'])
y_test = le.transform(test_articles['category'])

# Train a classifier (e.g., Logistic Regression)
st.write("Training Logistic Regression model...")
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Create a function to predict new inputs
def predict_category(text, model, vectorizer, label_encoder):
    clean_text = remove_punctuation_numbers(text.lower())  # Clean the input text
    transformed_text = vectorizer.transform([clean_text])  # Vectorize the text
    prediction = model.predict(transformed_text)  # Make a prediction
    return label_encoder.inverse_transform(prediction)[0]  # Decode the label

# User Input Section
st.write("## Predict Article Category")
user_input = st.text_area("Enter the text of a news article:", "")

if st.button('Predict Category'):
    if user_input:
        predicted_category = predict_category(user_input, clf, vect, le)
        st.write(f"The predicted category is: **{predicted_category}**")
    else:
        st.write("Please enter some text to classify.")

# Model Evaluation (if desired, you can keep this section as is)
st.write("## Model Performance")
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
test_accuracy = metrics.accuracy_score(y_test, y_pred_test)

st.write(f"Training accuracy: {train_accuracy}")
st.write(f"Test accuracy: {test_accuracy}")
