from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pickle
import ngram_processor


def process_ngrams(file_name):
    """
    Preprocess n-grams in the provided file.
    
    :param file_name: Name of the file to process.
    """
    ngram_processor.process_file(file_name)

def load_ngrams(file_name, label):
    """
    Load n-grams from a file and assign labels.
    
    :param file_name: Name of the file to load.
    :param label: Label for the n-grams in the file.
    :return: Tuple of n-grams and their corresponding labels.
    """
    with open(file_name, 'r') as f:
        ngrams = f.readlines()

    labels = [label] * len(ngrams)
    return ngrams, labels


def main():
    # Preprocess data files
    process_ngrams("Data/genre-related.txt")
    process_ngrams("Data/not-genre-related.txt")

    # Load and concatenate n-gram data
    genre_ngrams, genre_labels = load_ngrams("Data/genre-related.txt", 1)
    not_genre_ngrams, not_genre_labels = load_ngrams("Data/not-genre-related.txt", 0)
    all_ngrams = genre_ngrams + not_genre_ngrams
    all_labels = genre_labels + not_genre_labels

    # Perform stratified split on the data
    x_train, x_test, y_train, y_test = train_test_split(all_ngrams, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    # Define processing pipeline and hyperparameter search grid
    ngram_range = (1, 2)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=ngram_range)),
        ('clf', LogisticRegression())
    ])

    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'clf__penalty': ['l2']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)

    # Train the model using grid search
    grid_search.fit(x_train, y_train)

    # Evaluate the best model on the test data
    y_pred = grid_search.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=["not-genre-related", "genre-related"]))

    # Save the best model
    with open("Artifacts/pipeline.pkl", "wb") as best_pipeline_file:
        pickle.dump(grid_search.best_estimator_, best_pipeline_file)

if __name__ == "__main__":
    main()
