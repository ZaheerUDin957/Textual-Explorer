import streamlit as st
import re
import string
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
import spacy
import nltk
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

# Cleaned corpus as provided
corpus = [
    "artificial", "intelligence", "ai", "machine", "learning", "rvolutionizing", "teach", "focused", "also", 
    "learn", "data", "ai", "includes", "nap", "computer", "vision", "robotics", "learn", "neutral", "nets", 
    "developed", "gained", "prominence", "must", "century", "many", "companies", "integrate", "ai", "systems", 
    "selfdriving", "cars", "use", "ai", "also", "overfitting", "common", "issue", "nt", "forget", "biasvariance", 
    "tradeoff", "hyperparameter", "tuning", "crucial", "data", "new", "oil", "relief", "vast", "datasets", 
    "regularization", "like", "improves", "models", "tensorflow", "porch", "spiritlearn", "popular", "libraries", 
    "start", "online", "ai", "courses", "preprocessing", "key", "clean", "data", "better", "models", "removing", 
    "outlines", "handling", "missing", "values", "essential", "feature", "engineering", "boots", "model", "accuracy", 
    "pa", "tshe", "help", "large", "datasets", "crossvariation", "ensures", "reliable", "models", "accuracy", 
    "nt", "everything", "consider", "precision", "recall", "generalization", "unseen", "data", "tough", "models", 
    "interpretable", "others", "black", "boxes", "ethics", "ai", "design", "crucial", "watch", "algorithmic", 
    "bias", "deep", "learning", "often", "needs", "pus", "fast", "training", "cloud", "offers", "capable", 
    "ai", "infrastructure", "aimy", "used", "healthcare", "finance", "etc", "sentiment", "analysis", "via", 
    "nap", "common", "best", "aids", "various", "nap", "tasks", "transfer", "learning", "saves", "time", 
    "spelling", "errors", "types", "affect", "models", "stop", "words", "add", "noise", "tokenization", 
    "step", "text", "preprocessing", "labelled", "data", "vital", "supervised", "learning", "kmeans", "clusters", 
    "data", "model", "interpretability", "offers", "insight", "ai", "ethics", "growing", "huge", "implication", 
    "ai", "automatic", "may", "cause", "job", "displacement", "monitor", "production", "models", "drift", 
    "ensure", "gdp", "compliance", "ai", "boots", "efficiency", "adds", "risks", "bias", "training", "data", 
    "unfair", "ai", "ai", "potential", "limitless", "challenges", "persist", "models", "good", "training", 
    "data", "ai", "transform", "industries", "rapidly", "long", "number", "handle"
]

# 2. Bag of Words
def bag_of_words(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    st.subheader("Bag of Words")
    st.write(df.head())

# 3. N-grams (Unigram, Bigram, Trigram)
def ngrams(corpus, n=1):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    st.subheader(f"{n}-gram")
    st.write(df.head())

# 4. TF-IDF
def tfidf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    st.subheader("TF-IDF")
    st.write(df.head())

# 5. Word2Vec
def word2vec(corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0)
    vectors = {word: model.wv[word] for word in model.wv.index_to_key}
    df = pd.DataFrame(vectors).T
    st.subheader("Word2Vec (CBOW)")
    st.write(df.head())

# 6. SkipGram and CBOW
def word2vec_models(corpus, sg=0):
    tokenized_corpus = [doc.split() for doc in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=sg)
    vectors = {word: model.wv[word] for word in model.wv.index_to_key}
    df = pd.DataFrame(vectors).T  # Define df here with the vectors
    model_type = "SkipGram" if sg == 1 else "CBOW"
    st.subheader(f"Word2Vec ({model_type})")
    st.write(df.head())


# 7. GloVe
def glove(corpus):
    glove_vectors = api.load('glove-wiki-gigaword-50')
    vectors = {word: glove_vectors[word] for word in corpus if word in glove_vectors}
    df = pd.DataFrame(vectors).T
    st.subheader("GloVe")
    st.write(df.head())

# Main function to call all feature engineering techniques
def feature_engineering():
    st.title("Feature Engineering Techniques")
    bag_of_words(corpus)
    ngrams(corpus, n=1)  # Unigram
    # ngrams(corpus, n=2)  # Bigram
    # ngrams(corpus, n=3)  # Trigram
    tfidf(corpus)
    word2vec(corpus)
    word2vec_models(corpus, sg=1)  # SkipGram
    glove(corpus)


# Function for lowercasing
def to_lowercase(text):
    return text.lower()

# Function for uppercasing
def to_uppercase(text):
    return text.upper()

# Function to remove HTML tags
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Function to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+', '', text)


# Function to remove punctuation from a list of words
def remove_punctuation(word_list):
    table = str.maketrans('', '', string.punctuation)
    return [word.translate(table) for word in word_list if word.translate(table)]


# Function for chat word treatment (short word to normal data)
chat_words_dict = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "thx": "thanks",
    "pls": "please",
    "gr8": "great",
    "4u": "for you",
    "idk": "I don't know",
    "imo": "in my opinion",
    "btw": "by the way"
}

def chat_word_treatment(text):
    words = text.split()
    treated_words = [chat_words_dict.get(word, word) for word in words]
    return ' '.join(treated_words)

# Function for spelling correction
def correct_spelling(text):
    return str(TextBlob(text).correct())

# Function to remove stop words
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    removed_stopwords = [word for word in words if word.lower() in stop_words]
    return ' '.join(filtered_words), removed_stopwords, len(words) - len(filtered_words)

# Function to remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Function to delete emojis
def delete_emojis(text):
    return emoji.replace_emoji(text, "")

# Function to replace emojis with meaning
def replace_emojis_with_meaning(text):
    return emoji.demojize(text)

# Function to convert emojis to unicode
def convert_emojis_to_unicode(text):
    return text.encode('unicode_escape').decode('ASCII')

# Function for sentence-based tokenization
def sentence_tokenization(text):
    return sent_tokenize(text)

# Function for word-based tokenization
def word_tokenization(text):
    return word_tokenize(text)

# Function for POS tagging
def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# Function for parsing
def parsing(text):
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

# Functions for stemming using different algorithms
def porter_stemming(text):
    ps = PorterStemmer()
    words = word_tokenization(text)
    return [ps.stem(word) for word in words]

def lancaster_stemming(text):
    ls = LancasterStemmer()
    words = word_tokenization(text)
    return [ls.stem(word) for word in words]

def snowball_stemming(text):
    ss = SnowballStemmer("english")
    words = word_tokenization(text)
    return [ss.stem(word) for word in words]

# Functions for lemmatization using different algorithms
def wordnet_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenization(text)
    return [lemmatizer.lemmatize(word) for word in words]

def spacy_lemmatization(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def analyze_and_visualize_text(text_list):
    # Combine all text entries into a single string
    text = ' '.join(text_list)
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Word Frequency Distribution
    word_freq = Counter(filtered_tokens)
    
    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # Sentiment Analysis
    sentiment = TextBlob(text).sentiment
    
    # Part-of-Speech Tagging using spaCy
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    pos_freq = Counter(pos_tags)
    
    # Word Length Distribution
    word_lengths = [len(word) for word in filtered_tokens]
    
    # Streamlit Visualization
    st.title("Text Analysis and Visualization")

    # Word Frequency Distribution
    st.subheader("Word Frequency Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get the 20 most frequent words
    most_common_words = dict(Counter(filtered_tokens).most_common(20))

    ax.bar(most_common_words.keys(), most_common_words.values())
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Word Cloud
    st.subheader("Word Cloud")
    st.image(wordcloud.to_array(), use_column_width=True)

    # Sentiment Analysis (Pie Chart)
    st.subheader("Sentiment Analysis")
    sentiment_labels = ['Positive', 'Neutral', 'Negative']
    sentiment_values = [
        sum(1 for s in [sentiment.polarity] if s > 0),
        sum(1 for s in [sentiment.polarity] if s == 0),
        sum(1 for s in [sentiment.polarity] if s < 0)
    ]
    fig, ax = plt.subplots()
    ax.pie(sentiment_values, labels=sentiment_labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    ax.set_title('Sentiment Distribution')
    st.pyplot(fig)
    
    # Part-of-Speech Tagging (Bar Chart)
    st.subheader("Part-of-Speech Tagging")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(pos_freq.keys()), y=list(pos_freq.values()), hue=list(pos_freq.keys()), dodge=False, palette='viridis', ax=ax, legend=False)
    ax.set_xlabel('Part-of-Speech Tags')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Word Length Distribution (Histogram)
    st.subheader("Word Length Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(word_lengths, bins=range(1, max(word_lengths) + 1), edgecolor='black')
    ax.set_xlabel('Word Length')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Text Statistics
    num_words = len(filtered_tokens)
    num_unique_words = len(set(filtered_tokens))
    
    st.subheader("Text Statistics")
    st.write(f"Total Words: {num_words}")
    st.write(f"Unique Words: {num_unique_words}")


def show():
    st.set_page_config(page_title="NLP Project", layout="wide")
    dirty_corpus = """
    Artificial Intelligence (AI) & Machine Learning (ML) are rvolutionizing tech! <br> ML focuses on algos that "learn" from data! 🌟 AI includes NLP, computer vision & robotics. Learn more <a href='https://ai.example.com'>here</a>. Neural nets, developed in the 50s, gained prominence in the 21st century. Many companies integrate AI into systems 🤖. Self-driving cars use AI algos! Overfitting is a common issue 😅. Don't forget the bias-variance trade-off. Hyperparameter tuning is crucial! Data is the new oil, and ML relies on vast datasets 💾. Regularization like L2 improves models. TensorFlow, PyTorch, & scikit-learn are popular libraries. U can start with online AI courses. Pre-processing is key; clean data = better models! Removing outliers & handling missing values is essential. Feature engineering boosts model accuracy. PCA & t-SNE help with large datasets. Cross-validation ensures reliable models. Accuracy isn't everything; consider precision & recall too! Generalization to unseen data is tough. Some models are interpretable; others are black boxes. Ethics in AI design is crucial 💡. Watch out for algorithmic bias. Deep learning often needs GPUs for fast training. The cloud offers scalable AI infrastructure ☁️. AI/ML are used in healthcare, finance, etc. Sentiment analysis via NLP is common. BERT aids in various NLP tasks. Transfer learning saves time! Spelling errors & typ0s affect models 😬. Stop words add noise. Tokenization is step 1 in text pre-processing. Labeled data is vital for supervised learning. K-means clusters data. Model interpretability offers insights. AI ethics is growing with huge implications. AI automation may cause job displacement. Monitor production models for drift. Ensure GDPR compliance. AI boosts efficiency but adds risks. Bias in training data = unfair AI. AI's potential is limitless, but challenges persist. ML models are only as good as their training data. AI transforms industries rapidly! 1234567890 is a long number to handle.
    """
    st.title("Text Preprocessing Pipeline")
    st.header("Original Dirty Corpus")
    st.write(dirty_corpus)
    st.header("Text Cleaning")

    st.header("1. Lowercasing and Uppercasing")
    lowercased_text = to_lowercase(dirty_corpus)
    uppercased_text = to_uppercase(dirty_corpus)
    st.subheader("Lowercased Corpus:")
    st.write(lowercased_text)
    st.subheader("Uppercased Corpus:")
    st.write(uppercased_text)

    st.header("2. Remove HTML Tags")
    html_removed_text = remove_html_tags(lowercased_text)
    st.write(html_removed_text)

    st.header("3. Remove URLs")
    urls_removed_text = remove_urls(html_removed_text)
    st.write(urls_removed_text)

    st.header("4. Chat Word Treatment")
    st.subheader('chat_words_dictionary')
    st.write(chat_words_dict)
    chat_words_treated_text = chat_word_treatment(html_removed_text)
    st.write(chat_words_treated_text)

    st.header("5. Spelling Correction")
    spelling_corrected_text = correct_spelling(chat_words_treated_text)
    st.write(spelling_corrected_text)

    st.header("6. Remove Stop Words")
    stopwords_removed_text, removed_stopwords, num_stopwords_removed = remove_stopwords(spelling_corrected_text)

    # Print the results
    st.write("Filtered text:", stopwords_removed_text)
    st.write("Stopwords removed:", removed_stopwords)
    st.write("Number of stopwords removed:", num_stopwords_removed)

    st.header("7. Remove Numbers")
    numbers_removed_text = remove_numbers(stopwords_removed_text)
    st.write(numbers_removed_text)

    st.header("8. Emoji Handling")
    emojis_deleted_text = delete_emojis(numbers_removed_text)
    emojis_replaced_text = replace_emojis_with_meaning(numbers_removed_text)
    emojis_unicode_text = convert_emojis_to_unicode(numbers_removed_text)
    st.subheader("Emojis Deleted:")
    st.write(emojis_deleted_text)
    st.subheader("Emojis Replaced with Meaning:")
    st.write(emojis_replaced_text)
    st.subheader("Emojis Converted to Unicode:")
    st.write(emojis_unicode_text)

    st.header("9. POS Tagging")
    pos_tags = pos_tagging(emojis_deleted_text)
    st.subheader("POS Tags:")
    st.write(pos_tags)

    st.header("10. Parsing")
    parsing_result = parsing(emojis_deleted_text)
    st.write("Parsing Result:")
    st.write(parsing_result)

    st.header("11. Stemming")
    porter_stemmed = porter_stemming(emojis_deleted_text)
    lancaster_stemmed = lancaster_stemming(emojis_deleted_text)
    snowball_stemmed = snowball_stemming(emojis_deleted_text)
    st.write("Porter Stemming:")
    st.write(porter_stemmed)
    st.write("Lancaster Stemming:")
    st.write(lancaster_stemmed)
    st.write("Snowball Stemming:")
    st.write(snowball_stemmed)

    # Apply lemmatization
    st.header("12. Lemmatization")
    wordnet_lemmatized = wordnet_lemmatization(emojis_deleted_text)
    spacy_lemmatized = spacy_lemmatization(emojis_deleted_text)
    st.write("WordNet Lemmatization:")
    st.write(wordnet_lemmatized)
    st.write("spaCy Lemmatization:")
    st.write(spacy_lemmatized)

    st.header("13. Tokenization")
    sentences = sentence_tokenization(emojis_deleted_text)
    words = word_tokenization(emojis_deleted_text)
    st.write("Sentence Tokenization:")
    st.write(sentences)
    st.write("Word Tokenization:")
    st.write(words)

    # Apply lowercasing, remove HTML tags, remove URLs, and remove punctuation
    st.header("14. Remove Punctuation")
    punctuation_removed_text = remove_punctuation(words)
    st.write(punctuation_removed_text)

    Clean_corpus = punctuation_removed_text
    st.subheader('Cleaned Corpus')
    st.write(Clean_corpus)

    analyze_and_visualize_text(Clean_corpus)

    st.header('Textual Data Feature Engineering')
    feature_engineering()


show()
