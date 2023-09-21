import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download('punkt')

def load_ngrams(file_name, label):
    """
    Load the n-grams file
    :param file_name: The file to load
    :param label: The label for the n-grams in this file (0 or 1)
    :return: ngrams and labels
    """
    with open(file_name, 'r') as f:
        ngrams = f.readlines()
    labels = [label] * len(ngrams)
    return ngrams, labels

def main():
    # Load data
    genre_ngrams, genre_labels = load_ngrams("Data/genre-related.txt", 1)
    not_genre_ngrams, not_genre_labels = load_ngrams("Data/not-genre-related.txt", 0)

    all_ngrams = genre_ngrams + not_genre_ngrams
    all_labels = genre_labels + not_genre_labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_ngrams, all_labels, test_size=0.2, random_state=42)

    # Convert n-grams into TF-IDF vectors
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english', lowercase=True)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train the classifier
    classifier = LogisticRegression()
    classifier.fit(X_train_vect, y_train)

    # Test the classifier
    y_pred = classifier.predict(X_test_vect)
    print(classification_report(y_test, y_pred, target_names=["not-genre-related", "genre-related"]))

    # Save the model and vectorizer for future use
    with open("Artifacts/classifier.pkl", "wb") as model_file:
        pickle.dump(classifier, model_file)

    with open("Artifacts/vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

if __name__ == "__main__":
    main()
