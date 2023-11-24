# NLP_Assignment_1
Natural Language Processing first assignment - Text classification with Naive Bayes

## Dependencies
The project uses the following dependencies: 
1. NLTK
2. NumPy
3. Requests

## Project pipeline
The project is composed of 3 Python files:
1. **corpus_creation.py** -> creates the corpus by requesting documents through 
[Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page). The documents are automatically divided into training and
test set
2. **binary_classifier.py** -> contains a class named BinaryClassifier that, given the directory containing the training
documents, processes each text and builds a Naive Bayes classifier. The text processing pipeline is the following:
    * _Text normalization_:
      1. Punctuation removal
      2. Tokenization
      3. Stop words removal
      4. Stemming
    * _Bag of words creation_: for each class of documents (_medical_ and _non medical_) a bag of words is created
    * _Vocabulary creation_: the class bag of words are merged in order to create a vocabulary of the training documents
    * _Naive Bayes classifier training_
3. **main.py** -> runs a test of the created classifier