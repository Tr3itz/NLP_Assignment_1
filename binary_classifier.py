import re
import string
import numpy as np

from glob import glob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Remove punctuation through regular expression
def remove_punctuation(text: str) -> str:
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub("", text)


# Text normalization pipeline
def text_normalization(text: str) -> list[str]:
    stop_list = stopwords.words('english')
    ps = PorterStemmer()

    # Punctuation removal
    text = remove_punctuation(text)

    # Tokenization -> removing tokens of length 1 such as single characters or other residues
    tokens = [token.lower() for token in word_tokenize(text) if len(token) > 1]

    # Stop words removal
    tokens = [t for t in tokens if t not in stop_list]

    # Stemming
    stems = [ps.stem(t) for t in tokens]

    return stems


# Creation of a bag of words given a set of files
def bow_creation(files: list[str]) -> dict:
    bow: dict = {}

    for doc in files:
        with open(doc, 'r+') as f:
            stems: list[str] = text_normalization(f.read())

            for s in stems:
                if s not in bow.keys():
                    bow[s] = 1
                else:
                    bow[s] += 1

    return bow


# Computing the likelihood of each word within the vocabulary of appearing in each class
def class_loglikelihood(class_bow: dict, vocabulary: dict) -> dict:
    total_words = len(vocabulary.keys())
    total_occurrences = sum(vocabulary.values())

    # dictionary in which
    # keys: words within the vocabulary
    # values: loglikelihood of each word of belonging to the provided class
    loglikelihood = {}

    for word in vocabulary.keys():
        # Occurrences of the word within the documents of a class
        class_occurrences = 0 if word not in class_bow.keys() else class_bow[word]

        # The likelihood is smoothed with Laplace smoothing
        loglikelihood[word] = np.log((class_occurrences + 1) / (total_occurrences + total_words))

    return loglikelihood


class BinaryClassifier:
    """
    TRAINING OF THE MODEL
    Creates the classifier object by creating bag of words each class of documents and merging them
    into one vocabulary of all documents in the training set

    @:param training_directory -> directory containing both medical and non-medical files
    """
    def __init__(self, training_directory: str):

        print('Retrieving training documents...')

        # Training documents
        training_medical_documents = glob(f'{training_directory}/medical/*.txt')
        training_other_documents = glob(f'{training_directory}/non-medical/*.txt')

        print('Creation of the vocabulary...')

        # Bag of words of single classes
        self.medical_bow = bow_creation(training_medical_documents)
        self.other_bow = bow_creation(training_other_documents)

        # Vocabulary of all the documents
        self.vocabulary = self.medical_bow.copy()
        for word in self.other_bow.keys():
            if word not in self.vocabulary.keys():
                self.vocabulary[word] = self.other_bow[word]
            else:
                self.vocabulary[word] += self.other_bow[word]

        print('Computing class probabilities...')

        # Training a Naive Bayes classifier
        self.medical_prior = np.log(len(training_medical_documents)
                                    / (len(training_medical_documents) + len(training_other_documents)))

        self.other_prior = np.log(len(training_other_documents)
                                  / (len(training_medical_documents) + len(training_other_documents)))

        print(f'Prior log probability of medical documents: {self.medical_prior}')
        print(f'Prior log probability of non-medical documents: {self.other_prior}')

        self.medical_loglikelihood = class_loglikelihood(self.medical_bow, self.vocabulary)
        self.other_loglikelihood = class_loglikelihood(self.other_bow, self.vocabulary)

    """ 
    TESTING THE MODEL
    It classifies all the documents inside the provided directory. The classification is performed 
    using the bag of words and likelihoods of each class and the vocabulary of all the training documents.
    
    @:param test_directory -> directory containing all the test documents on which the classification has to be
                              performed
    
    @:return system_label -> list of predictions on each document performed by the classifier
    @:return gold_labels -> list of labels that indicate the actual class each document belongs to
    """
    def classification(self, test_directory: str) -> tuple:

        print('\nRetrieving test documents...')

        # Test files retrieval
        test_medical_files = glob(f'{test_directory}/medical/*.txt')
        test_other_files = glob(f'{test_directory}/non-medical/*.txt')
        test_files = test_medical_files + test_other_files

        # Actual labels of the test files
        gold_labels = np.concatenate([np.zeros(len(test_medical_files)), np.ones(len(test_other_files))])

        # Labels given to the test files by the classifier
        system_labels = np.zeros(len(test_files))

        print('Documents classification...')

        for i in range(len(test_files)):
            # List of posterior probabilities
            posterior = [self.medical_prior, self.other_prior]

            # Classifying each test file with the class that returns the highest posterior probability
            with open(test_files[i], 'r+') as f:
                words = text_normalization(f.read())

                # The posterior probability is given by the summation
                # of the likelihoods of each word to occur in the class
                for w in words:
                    # We consider only the words appearing in our vocabulary
                    if w in self.vocabulary.keys():
                        posterior[0] += self.medical_loglikelihood[w]
                        posterior[1] += self.other_loglikelihood[w]
                system_labels[i] = np.argmax(posterior)

        return system_labels, gold_labels
