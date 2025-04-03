import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import streamlit as st

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # Download the 'punkt_tab' resource

data = pd.read_csv("train.csv", encoding='latin-1')
data.fillna('', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.2, random_state=42)



# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and links
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

X_train['preprocessed_text'] = X_train['text'].apply(preprocess_text)

X_test['preprocessed_text'] = X_test['text'].apply(preprocess_text)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the class labels in both training and validation datasets
train_labels_encoded = label_encoder.fit_transform(X_train['sentiment'])
validation_labels_encoded = label_encoder.transform(X_test['sentiment'])

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the 'max_features' parameter based on your dataset

# Fit and transform the training data
tfidf_train_features = tfidf_vectorizer.fit_transform(X_train['preprocessed_text'])

with open('tfidf_vectorizer.pickle', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Transform the validation data using the same vectorizer
tfidf_validation_features = tfidf_vectorizer.transform(X_test['preprocessed_text'])

# Display the shape of the TF-IDF features
print(f"TF-IDF Training Features Shape: {tfidf_train_features.shape}")
print(f"TF-IDF Validation Features Shape: {tfidf_validation_features.shape}")

# Initialize the XGBoost classifier
xgb_classifier = xgb.XGBClassifier()

# Train the classifier on the TF-IDF training features and encoded labels
xgb_classifier.fit(tfidf_train_features, train_labels_encoded)

# Predict the encoded labels for the TF-IDF validation features
validation_predictions_xgb_encoded = xgb_classifier.predict(tfidf_validation_features)

# Decode the predicted labels back to the original class labels
validation_predictions_xgb = label_encoder.inverse_transform(validation_predictions_xgb_encoded)

# Evaluate the performance of the XGBoost classifier
accuracy_xgb = accuracy_score(X_test['sentiment'], validation_predictions_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

# Streamlit app
st.title("Sentiment Analysis")

# Input text area
user_input = st.text_area("Enter your comment:")

# Predict button
if st.button("Predict"):
    # Preprocess the input text
    preprocessed_input = preprocess_text(user_input)

    # Vectorize the preprocessed input
    input_vector = tfidf_vectorizer .transform([preprocessed_input])

    # Make prediction
    prediction = xgb_classifier.predict(input_vector)[0]

    # Display the prediction
    if prediction == 0:
        st.write("Negative Sentiment")
    elif prediction == 1:
        st.write("Neutral Sentiment")
    else:
        st.write("Positive Sentiment")


