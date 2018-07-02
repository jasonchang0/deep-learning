import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

"""
[chair, table, spoon, television] ->
I pulled the chair up to the table ->
[1, 1, 0, 0]
"""

lemmatizer = WordNetLemmatizer()
num_lines = int(1e7)


def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for l in contents[:num_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(_) for _ in lexicon]
    w_counts = Counter(lexicon)
    # w_counts = {'the': 52521, 'and': 25242}

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            # keep the magnitude of neural network in tact
            l2.append(w)

    print('Length of Lexicon:', len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
    feature_set = []
    # [
    # [[0 1 0 1 1 0], [0 1]] ->
    # [[words], [pos, neg]]
    # ]

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:num_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(_) for _ in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_val = lexicon.index(word.lower())
                    features[index_val] += 1

            features = list(features)
            feature_set.append([features, classification])

    return feature_set


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features.extend(sample_handling('pos.txt', lexicon, [1, 0]))
    features.extend(sample_handling('neg.txt', lexicon, [0, 1]))
    random.shuffle(features)

    """
    does tf.argmax([output]) == tf.argmax([expectations])?
    does tf.argmax([52412, 23421]) == tf.argmax([1, 0])?
    """

    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as file:
        pickle.dump([train_x, train_y, test_x, test_y], file)





