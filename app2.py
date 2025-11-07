import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from docx import Document # Import Document for docx file reading

# Load the saved TF-IDF vectorizer and the trained model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('best_decision_tree_model.pkl')

# Define the text cleaning and lemmatization functions (same as used during training)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

def extract_text_from_docx(file):
    """
    Extracts text content from an uploaded .docx file.

    Args:
        file (file-like object): The uploaded .docx file.

    Returns:
        str: The extracted text content from the file.
    """
    document = Document(file)
    text = ''
    for paragraph in document.paragraphs:
        text += paragraph.text + '\n'
    return text

# Create the Streamlit application
st.title("Resume Category Predictor")

st.write("Upload a resume file (.docx) to predict its category.")

# Add file uploader
uploaded_file = st.file_uploader("Upload Resume File", type=["docx"])

if uploaded_file is not None:
    # Extract text from the uploaded file
    resume_text = extract_text_from_docx(uploaded_file)

    if st.button("Predict Category"):
        if resume_text:
            # Preprocess the extracted text
            cleaned_text = clean_text(resume_text)
            lemmatized_text = lemmatize_text(cleaned_text)

            # Transform the preprocessed text using the loaded vectorizer
            text_tfidf = vectorizer.transform([lemmatized_text])

            # Make the prediction
            prediction = model.predict(text_tfidf)

            # Decode the prediction (assuming you have the original label encoder or a mapping)
            # For simplicity, let's use a hardcoded mapping based on your earlier value counts:
            category_mapping = {
                0: 'Intern Resumes',
                1: 'Peoplesoft Resumes',
                2: 'React Js Resumes',
                3: 'SQL Resumes',
                4: 'Workday Resumes'
            }
            predicted_category = category_mapping.get(prediction[0], "Unknown Category")

            st.success(f"Predicted Category: {predicted_category}")
        else:
            st.warning("Could not extract text from the uploaded file.")
else:
    st.info("Please upload a .docx file to get a prediction.")
