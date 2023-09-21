# game genre-related n-gram classifier
This is just a really quick classifier to look at n-grams in a game's description and determine if they are likely or unlikely to be relevant for game genre determination. 
Useful as a text pre-processing step to filter out irrelevant text from game descriptions when you want something more comprehensive than stop words.
Also a straightforward example of building a little binary classifier with limited data.
Data in genre-related.txt and non-genre-related.txt was synthesized with chat-gpt. If you make changes, run `python pre-process-n-grams.py` to ensure that entries are formatted correctly.
Run `python train-classifier.py` -- I've not included the pickled output because you really shouldn't be running stuff that you didn't pickle yourself.