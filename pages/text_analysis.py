import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

@st.cache_data
def load_data():
    df = pd.read_csv("clothing.csv")
    return df

df = load_data()

def preprocess_text(text):
    #tokenization
    tokens = word_tokenize(text)
    #stopwords_removal
    stop_words = set(stopwords.words('english'))
    #converting_lowecase
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return filtered_tokens

def normalize_text(tokens):
    #stemming
    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(token) for token in tokens]
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    return lemmatized_tokens

# Text Similarity Analysis
def similarity_analysis(df, division):
    df['Division Name'] = df['Division Name'].astype(str)
    division_df = df[df['Division Name'] == division]
    division_df['Review Text'] = division_df['Review Text'].astype(str)

    # Preprocessing
    division_df['Processed Review'] = division_df['Review Text'].apply(preprocess_text)
    division_df['Normalized Review'] = division_df['Processed Review'].apply(normalize_text)

    if len(division_df) != len(division_df['Normalized Review'].dropna()):
        print("Warning: Dataframe size mismatch between processed and normalized reviews. Jaccard similarity might be inaccurate.")
    processed_reviews = division_df['Normalized Review'].dropna().tolist()

    # TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_reviews)

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Jaccard similarity
    jaccard_sim = []
    for i in range(len(processed_reviews)):
        for j in range(i + 1, len(processed_reviews)):
            if processed_reviews[i] and processed_reviews[j]:
                jaccard_sim.append(jaccard_score(processed_reviews[i], processed_reviews[j], average='weighted'))

    return cosine_sim, jaccard_sim



st.title("Text Similarity Analysis")
division = st.selectbox("Select Division", df['Division Name'].unique())

if st.button("Run Similarity Analysis"):
    cosine_sim, jaccard_sim = similarity_analysis(df, division)
    st.write("Cosine Similarity Matrix:")
    st.write(pd.DataFrame(cosine_sim, columns=df[df['Division Name'] == division].index,
                           index=df[df['Division Name'] == division].index))
    st.write("Jaccard Similarity Scores:")
    st.write(jaccard_sim)