import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load existing model and vectorizer
with open('tfidf_vectorizer.pickle', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('xgb_classifier_model.pickle', 'rb') as f:
    xgb_classifier = pickle.load(f)

# Streamlit app
st.title("Sentiment Analysis with Retraining")

# Input text
user_input = st.text_area("Enter your text here:")

# Preprocess and predict
if user_input:
    processed_input = preprocess_text(user_input)
    input_tfidf = tfidf_vectorizer.transform([processed_input])
    prediction_encoded = xgb_classifier.predict(input_tfidf)[0]
    label_encoder = LabelEncoder()  # Initialize LabelEncoder
    label_encoder.classes_ = ['negative', 'neutral', 'positive']  # Assuming your classes
    predicted_type = label_encoder.inverse_transform([prediction_encoded])[0]
    st.write(f"Predicted sentiment: {predicted_type}")

# Retraining section
st.header("Retrain the model")
new_data = st.text_area("Enter new training data (text,sentiment):\n(One example per line, separated by comma)")

if st.button("Retrain"):
    if new_data:
        # Create DataFrame from new data
        new_data_list = [line.strip().split(',') for line in new_data.split('\n') if line.strip()]
        new_df = pd.DataFrame(new_data_list, columns=['text', 'sentiment'])

        # Preprocess new data
        new_df['preprocessed_text'] = new_df['text'].apply(preprocess_text)

        # Encode labels
        label_encoder = LabelEncoder()
        new_labels_encoded = label_encoder.fit_transform(new_df['sentiment'])

        # Vectorize new data
        new_tfidf_features = tfidf_vectorizer.transform(new_df['preprocessed_text'])

        # Retrain the model
        xgb_classifier.fit(new_tfidf_features, new_labels_encoded)

        # Save the updated model (optional)
        with open('xgb_classifier_model.pickle', 'wb') as f:
            pickle.dump(xgb_classifier, f)

        st.success("Model retrained successfully!")
    else:
        st.warning("Please enter new training data.")