import scnn_utils
import numpy as np
import os
from SentimentCNN import SentimentCNN


def rt_movie_reviews_demo(test_ratio, embedding_size, filter_sizes,
    num_filters, batch_size, num_epochs, eval_at, keep_prob):
    """
    Runs the SentimentCNN on Rotten Tomatos Movie Review data.
    Source: www.cs.cornell.edu/people/pabo/movie-review-data/
    """
    
    positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
    negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"
    positive_examples = list(open(positive_data_file, "r").readlines())
    negative_examples = list(open(negative_data_file, "r").readlines())
    
    x_train, y_train, x_test, y_test, vocab_length = scnn_utils.classify_data(
        positive_examples, negative_examples, test_ratio)

    print("Vocabulary Size: {:d}".format(vocab_length))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

    cnn = SentimentCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=vocab_length,
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters
    )

    cnn.run(x_train, y_train, x_test, y_test, batch_size, num_epochs, eval_at, keep_prob)


def twitter_corpus_demo(num_inputs, test_ratio, embedding_size, filter_sizes,
    num_filters, batch_size, num_epochs, eval_at, keep_prob):
    """
    Runs a SentimentCNN on Twitter Data.
    Source: thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/
    """

    raw_data = list(open("./data/tweets/data.csv", "r").readlines())[1:num_inputs + 1]
    split_data = [sent.split(",") for sent in raw_data]
    positive_tweets = [s[3] for s in split_data if s[1] == "1"]
    negative_tweets = [s[3] for s in split_data if s[1] == "0"]

    x_train, y_train, x_test, y_test, vocab_length = scnn_utils.classify_data(
        positive_tweets, negative_tweets, test_ratio)

    print("Vocabulary Size: {:d}".format(vocab_length))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

    cnn = SentimentCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=vocab_length,
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters
    )

    cnn.run(x_train, y_train, x_test, y_test, batch_size, num_epochs, eval_at, keep_prob)


def stanford_movie_reviews_demo(test_ratio, embedding_size, filter_sizes,
    num_filters, batch_size, num_epochs, eval_at, keep_prob):
    """
    NOTE: Takes a large amount of memory, large vocabulary.

    Runs a SentimentCNN on Stanford's Movie Review data set.
    Source: ai.stanford.edu/~amaas/data/sentiment/
    """

    pos_dir = "./data/stanford_movie_reviews/pos"
    neg_dir = "./data/stanford_movie_reviews/neg"

    positive_data = []
    for f in os.listdir(pos_dir):
        positive_data.extend(list(open(os.path.join(pos_dir, f), "r").readlines()))

    negative_data = []
    for f in os.listdir(neg_dir):
        negative_data.extend(list(open(os.path.join(neg_dir, f), "r").readlines()))

    x_train, y_train, x_test, y_test, vocab_length = scnn_utils.classify_data(
            positive_data, negative_data, test_ratio)

    print("Vocabulary Size: {:d}".format(vocab_length))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

    cnn = SentimentCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=vocab_length,
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters
    )

    cnn.run(x_train, y_train, x_test, y_test, batch_size, num_epochs, eval_at, keep_prob)


if __name__ == "__main__":

    rt_movie_reviews_demo(
        test_ratio=0.2,
        embedding_size=128,
        filter_sizes=[3,4,5],
        num_filters=128,
        batch_size=50,
        num_epochs=200,
        eval_at=100,
        keep_prob=0.5
    )
