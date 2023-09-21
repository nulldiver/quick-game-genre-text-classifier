import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Download the required resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def process_ngrams(file_name):
    """
    Ensure that our n-grams are formatted correctly
    :param file_name: The file containing an n-gram list
    """
    # Load stop words
    stop_words = set(stopwords.words('english'))

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Read the file
    with open(file_name, 'r') as f:
        ngrams = f.readlines()

    # Process the ngrams
    processed_ngrams = []
    for ngram in ngrams:
        # Convert to lowercase
        ngram = ngram.lower()

        # Remove non alpha-numeric characters
        ngram = re.sub(r'[^a-z0-9\s]', '', ngram)

        # Tokenize
        tokens = word_tokenize(ngram)

        # Remove stop words and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        # Convert back to string
        ngram = ' '.join(tokens)

        # Exclude 1-grams and add to processed ngrams if it's not already in the list
        if len(tokens) > 1 and ngram not in processed_ngrams:
            processed_ngrams.append(ngram)

    # Save the processed ngrams back to the file
    with open(file_name, 'w') as f:
        for ngram in processed_ngrams:
            f.write(ngram + '\n')

if __name__ == "__main__":
    process_ngrams("Data/not-genre-related.txt")
    process_ngrams("Data/genre-related.txt")
