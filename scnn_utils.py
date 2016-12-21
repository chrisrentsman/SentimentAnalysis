import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn


def classify_data(positive_data, negative_data, test_ratio):
    """
    Takes sentiment analysis data and classifies them into positive
    and negative cases, and then splits them into training and test
    data.
    """
    
    # Strip data of unnecessary characters
    positive_data = [s.strip() for s in positive_data]
    negative_data = [s.strip() for s in negative_data]

    # Split by words
    x_text = positive_data + negative_data
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_data]
    negative_labels = [[1, 0] for _ in negative_data]
    y = np.concatenate([positive_labels, negative_labels], 0)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    test_data_cutoff = int(test_ratio * float(len(y)))
    x_train = x_shuffled[test_data_cutoff:]
    x_test = x_shuffled[:test_data_cutoff]
    y_train = y_shuffled[test_data_cutoff:]
    y_test = y_shuffled[:test_data_cutoff]

    return (x_train, y_train, x_test, y_test, len(vocab_processor.vocabulary_))


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def make_batches(data, batch_size, num_epochs, shuffle=True):
    """
    Turns data into batches.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):

        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


