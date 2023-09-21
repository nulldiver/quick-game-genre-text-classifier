# game genre-related n-gram classifier
This is just a really quick classifier to look at n-grams in a game's description and determine if they are likely or unlikely to be relevant for game genre determination. 

- Really just a simple TfidfVectorizer and LogisticRegression with GridSearchCV
- Useful as a text pre-processing step to filter out irrelevant text from game descriptions when you want something more comprehensive than stop words.
- Also a straightforward example of building a little binary classifier with limited data.
- Run `python train_classifier.py` -- I've not included the pickled output because you really shouldn't be running stuff that you didn't pickle yourself.

Results:

|                   | precision | recall | f1-score | support |
|-------------------|-----------|--------|----------|---------|
| non-genre-related | 0.95      | 0.95   | 0.95     | 93      |
| genre-related     | 0.93      | 0.93   | 0.93     | 75      |
| accuracy          |           |        | 0.94     | 168     |
| macro avg         | 0.94      | 0.94   | 0.94     | 168     |
| weighted avg      | 0.94      | 0.94   | 0.94     | 168     |
     

