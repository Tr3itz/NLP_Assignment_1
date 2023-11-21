import numpy as np

from binary_classifier import BinaryClassifier


def main():
    classifier = BinaryClassifier('./corpus/training')

    system_labels, gold_labels = classifier.classification('./corpus/test')

    # Confusion matrix
    confusion_matrix = np.zeros([2, 2])
    for i in range(len(system_labels)):
        confusion_matrix[int(system_labels[i])][int(gold_labels[i])] += 1

    # True positives, false positives, true negatives, false negatives
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    tn = confusion_matrix[1][1]
    fn = confusion_matrix[1][0]

    # Performance measures
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_1 = (2*precision*recall) / (precision + recall)

    print('\nRESULTS:')

    print(f'\nConfusion matrix:\n{confusion_matrix}\n')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 measure: {f_1}')


if __name__ == '__main__':
    main()
