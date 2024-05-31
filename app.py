# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


def get_user_session():
    if 'user' not in st.session_state:
        st.session_state.user = {'authenticated': False, 'username': None}
    return st.session_state.user

# Function to check if username exists
def username_exists(username):
    # Here, you can implement your logic to check if the username already exists in a database
    # For demonstration purposes, let's use a hardcoded list of existing usernames
    existing_usernames = ["user"]
    return username in existing_usernames

# Signup function
def signup(username, password):
    # Here, you can implement your logic to create a new user account in a database
    # For demonstration purposes, let's assume the signup is successful if the username is not already taken
    if not username_exists(username):
        user_session = get_user_session()
        user_session['authenticated'] = True
        user_session['username'] = username
        return True
    else:
        return False

# Login function
def login(username, password):
    # Here, you can implement your authentication logic, such as checking against a database
    # For demonstration purposes, let's use a hardcoded username and password
    if username == "user" and password == "password":
        user_session = get_user_session()
        user_session['authenticated'] = True
        user_session['username'] = username
        return True
    else:
        return False

# Logout function
def logout():
    user_session = get_user_session()
    user_session['authenticated'] = False
    user_session['username'] = None
    st.experimental_rerun()  # Reload the app


def save_models(lr_model, svm_model, rf_model):
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Get sentiment scores
    scores = sid.polarity_scores(text)
    
    # Classify sentiment based on compound score
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"
def fact_check(input_text):
    # Example: Check if the input contains the name of a world leader
    world_leaders = ["narendra modi", "donald trump", "angela merkel"]  # Add more leaders as needed
    for leader in world_leaders:
        if leader.lower() in input_text:
            return "Real"  # Return "Real" for factual statements about world leaders
    return None  # Return None if no factual statement is detected
# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load the dataset
@st.cache_data()
def load_data():
    df = pd.read_csv('fake_and_real_news_dataset.csv')
    return df

# Preprocess the dataset
@st.cache_data()
def preprocess_data(df):
    df['text'] = df['text'].apply(preprocess_text)
    return df

# Vectorize text using TF-IDF
def vectorize_text(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    return X, vectorizer

# Load models

@st.cache_resource()
def load_models():
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    return lr_model, svm_model, rf_model

# Main function
def main():

    user_session = get_user_session()
    st.title("Social Media Text and Sentiment Analyser")
    st.subheader("Predict whether a news is fake or real and perform sentiment analysis")
    
    user_session = get_user_session()
    
    if not user_session['authenticated']:
        st.subheader("Login / Signup")
        signup_mode = st.radio("New User? Signup here:", ("Login", "Signup"))
        if signup_mode == "Signup":
            new_username = st.text_input("Enter desired username:")
            new_password = st.text_input("Enter password:", type="password")
            if st.button("Signup"):
                if signup(new_username, new_password):
                    st.success("Account created. Please login.")
                else:
                    st.error("Username already exists. Please choose a different username.")
        else:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if login(username, password):
                    st.success("Login successful")
                else:
                    st.error("Invalid username or password")
    else:
        st.subheader(f"Welcome, {user_session['username']}!")
        if st.button("Logout"):
            logout()
            st.success("Logout successful")
        # Load data
        df = load_data()
        
        df = preprocess_data(df)
        
        X, vectorizer = vectorize_text(df)
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lr_model, svm_model, rf_model = load_models()
        st.sidebar.subheader("Predict")
        text = st.sidebar.text_area("Enter News Article Here")
        text = preprocess_text(text)
        fact_result = fact_check(text) # remove this
        
        sentiment = analyze_sentiment(text)
        
        if fact_result:
            st.write("### Fact-Checking Result")
            st.write(f"The statement '{text}' is classified as: {fact_result}")
            st.write("### Sentiment Analysis Result")
            st.write(f"The sentiment of the input text is: {sentiment}")
        else:
            if st.sidebar.button("Predict"):
                if text:
                    text = preprocess_text(text)
                    text_vectorized = vectorizer.transform([text])
                    lr_prediction = lr_model.predict(text_vectorized)
                    svm_prediction = svm_model.predict(text_vectorized)
                    rf_prediction = rf_model.predict(text_vectorized)
                    
                    st.write("Logistic Regression Prediction:", lr_prediction[0])
                    st.write("Linear SVM Prediction:", svm_prediction[0])
                    st.write("Random Forest Prediction:", rf_prediction[0])
                else:
                    st.warning("Please enter a news article to predict.")
        # st.subheader("Dataset")
        # st.write(df)
        st.subheader("Model Performance")
        st.write("Logistic Regression Accuracy:", accuracy_score(y_test, lr_model.predict(X_test)))
        st.write("Linear SVM Accuracy:", accuracy_score(y_test, svm_model.predict(X_test)))
        st.write("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))
    

if __name__ == "__main__":
    main()
