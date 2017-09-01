# -*- coding: utf-8 -*-
"""Creates sentiment featuresets by preprocessing the data with nltk."""
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


def create_lexicon(pos, neg):
    """Create Lexicon."""
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    resultlexicon = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:  # No super common words or too rare
            resultlexicon.append(w)
    return resultlexicon


def sample_handling(sample, lexicon, classification):
    """Handle samples and return a suitable format."""
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    """Create featuresets and labels (training and testing data)."""
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('positive.txt', lexicon, [1, 0])
    features += sample_handling('negative.txt', lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    # create training data.
    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    # create testing data.
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    p = 'positive.txt'
    n = 'negative.txt'
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(p, n)
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
