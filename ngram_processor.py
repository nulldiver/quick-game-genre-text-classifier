# ngram_processor.py

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Download the required resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load stop words
stop_words = set(stopwords.words('english'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def process_ngrams(content):
    """
    Ensure that our n-grams are formatted correctly
    :param content: A string containing an n-gram list
    :return: List of processed n-grams
    """
    # Split content into lines (ngrams)
    ngrams = content.splitlines()

    processed_ngrams = []
    for ngram in ngrams:
        ngram = ngram.lower()
        ngram = re.sub(r'[^a-z0-9\s]', '', ngram)
        tokens = word_tokenize(ngram)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        ngram = ' '.join(tokens)

        if len(tokens) > 1 and ngram not in processed_ngrams:
            processed_ngrams.append(ngram)

    return processed_ngrams

def process_file(file_name):
    """
    Process an n-gram file
    :param file_name: The file containing an n-gram list
    """
    with open(file_name, 'r') as f:
        content = f.read()

    processed_ngrams = process_ngrams(content)

    with open(file_name, 'w') as f:
        for ngram in processed_ngrams:
            f.write(ngram + '\n')


