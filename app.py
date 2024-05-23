import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

@st.cache_data(allow_output_mutation=True)
def load_data():
    fake_data = pd.read_csv('Fake.csv')
    real_data = pd.read_csv('True.csv')
    fake_data['label'] = 1
    real_data['label'] = 0
    data = pd.concat([fake_data, real_data], ignore_index=True)
    return data

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def train_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    X_train_preprocessed = X_train.apply(preprocess_text)
    X_test_preprocessed = X_test.apply(preprocess_text)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    X_train_vect = tfidf_vectorizer.fit_transform(X_train_preprocessed)
    X_test_vect = tfidf_vectorizer.transform(X_test_preprocessed)
    rf_classifier = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1, random_state=42)
    rf_classifier.fit(X_train_vect, y_train)
    return rf_classifier, tfidf_vectorizer, X_test_vect, y_test

def evaluate_model(rf_classifier, X_test_vect, y_test):
    y_pred = rf_classifier.predict(X_test_vect)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return report, matrix

def main():
    st.title("Fake News Detection")
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose a function", ("Train Model", "Evaluate Model", "Predict News"))

    if option == "Train Model":
        st.write("Training the model...")
        rf_classifier, tfidf_vectorizer, X_test_vect, y_test = train_model()
        joblib.dump(rf_classifier, 'random_forest_model.pkl')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        st.write("Model trained and saved successfully!")

    elif option == "Evaluate Model":
        rf_classifier = joblib.load('random_forest_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        data = load_data()
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
        X_test_preprocessed = X_test.apply(preprocess_text)
        X_test_vect = tfidf_vectorizer.transform(X_test_preprocessed)
        report, matrix = evaluate_model(rf_classifier, X_test_vect, y_test)
        st.write("Classification Report:")
        st.text(report)
        st.write("Confusion Matrix:")
        st.write(matrix)

    elif option == "Predict News":
        rf_classifier = joblib.load('random_forest_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        sample_text = st.text_area("Enter a news article to classify:")
        if st.button("Predict"):
            sample_text_preprocessed = preprocess_text(sample_text)
            sample_input_vect = tfidf_vectorizer.transform([sample_text_preprocessed])
            predicted_label = rf_classifier.predict(sample_input_vect)
            if predicted_label == 1:
                st.write("The sample input is predicted to be **FAKE** news.")
            else:
                st.write("The sample input is predicted to be **REAL** news.")

if __name__ == '__main__':
    main()
