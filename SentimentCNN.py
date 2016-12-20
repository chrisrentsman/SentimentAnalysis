import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn

class SentimentCNN(object):
    """
    Constructs a Convolutional Neural Network for sentiment analysis.

    Performs an embedding layer, constructing word embeddings for all words
    in our vocabulary. Then, runs a convolution layer for each filter size,
    activates that output using a ReLU function, and then max pools the
    values.

    Afterwards, combines all of the pooled filters together into a one 
    dimensional vector for each input, and then guesses the class of
    the input.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, 
        embedding_size, filter_sizes, num_filters):

        # Placeholders for input, output, and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])

        # Embedding Layer
        W_embedded = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        self.embedded_chars = tf.nn.embedding_lookup(W_embedded, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Perform convolution and pooling for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]

            # Convolution layer
            W = self.weight_variable(filter_shape)
            b = self.bias_variable([num_filters])
            conv = self.convolution_step(self.embedded_chars_expanded, W)

            # Activation Layer
            h = tf.nn.relu(tf.nn.bias_add(conv, b))

            # Pooling
            pooled = self.max_pooling_step(h, [1, sequence_length - filter_size + 1, 1, 1])
            pooled_outputs.append(pooled)

        # Densely connected layer
        num_filters_total = num_filters * len(filter_sizes)
        h_pool_flat = tf.reshape(tf.concat(3, pooled_outputs), [-1, num_filters_total])

        # Add dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        # Guess outputs
        W_output = self.weight_variable([num_filters_total, num_classes])
        b_output = self.bias_variable([num_classes])
        scores = tf.nn.xw_plus_b(self.h_drop, W_output, b_output)
        predictions = tf.argmax(scores, 1)

        losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
        correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))

        self.loss = tf.reduce_mean(losses)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Initialize session and set up training operations
        self.sess = tf.InteractiveSession()
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train = self.optimizer.apply_gradients(self.grads_and_vars, 
            global_step=self.global_step)
        self.sess.run(tf.initialize_all_variables())

    def run(self, x_input, y_input, x_test, y_test, batch_size, num_epochs, eval_step, keep_prob):
        """Runs the entire neural network on given input."""

        batches = data_helpers.batch_iter(list(zip(x_input, y_input)), 
            batch_size, num_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_info = self.train_step(x_batch, y_batch, keep_prob)
            current_step = tf.train.global_step(self.sess, self.global_step)
            print("{}: step: {}, loss: {:g}, acc: {:g}".format(*train_info))

            if current_step % eval_step == 0: 
                print("\nEvaluation")
                eval_info = self.test_step(x_test, y_test)
                print("{}: step: {}, loss: {:g}, acc: {:g}\n".format(*eval_info))

    def train_step(self, x_batch, y_batch, keep_prob):
        """Trains the SentimentCNN with one batch of input and output."""

        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.keep_prob: keep_prob
        }

        _, step, loss, accuracy = self.sess.run(
            [self.train, self.global_step, self.loss, self.accuracy],
            feed_dict
        )

        current_time = datetime.datetime.now().isoformat()
        return (current_time, step, loss, accuracy)

    def test_step(self, x_batch, y_batch):
        """Tests the SentimentCNN with one batch of input and output."""

        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.keep_prob: 1.0
        }

        step, loss, accuracy = self.sess.run(
            [self.global_step, self.loss, self.accuracy],
            feed_dict
        )

        current_time = datetime.datetime.now().isoformat()
        return (current_time, step, loss, accuracy)

    def weight_variable(self, shape):
        """Initializes a weight variable with some noise."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Initializes a slightly positive bias variable."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def convolution_step(self, x, W):
        """Narrow convolution step with strides of one."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

    def max_pooling_step(self, x, dim):
        """Max pooling step with specified dimensions."""
        return tf.nn.max_pool(x, ksize=dim, strides=[1, 1, 1, 1], padding="VALID")


def movie_reviews(p_file, n_file, test_ratio, embedding_size, filter_sizes,
    num_filters, batch_size, num_epochs, eval_at, keep_prob):
    """
    Runs the SentimentCNN on Rotten Tomatos Movie Review data.
    Source: https://www.cs.cornell.edu/people/pabo/movie-review-data/
    """

    print("Loading movie review data...")
    x_text, y = data_helpers.load_data_and_labels(p_file, n_file)

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

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

    cnn = SentimentCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters
    )

    cnn.run(x_train, y_train, x_test, y_test, batch_size, num_epochs, eval_at, keep_prob)


def tweets(test_ratio, embedding_size, filter_sizes,
    num_filters, batch_size, num_epochs, eval_at, keep_prob):
    """Runs a SentimentCNN on Twitter Data."""

    # Split data in positive and negative tweets
    raw_data = list(open("./Sentiment Analysis Dataset.csv", "r").readlines())[1:10001]
    split_data = [sent.split(",") for sent in raw_data]
    positive_tweets = [s[3].strip() for s in split_data if s[1] == "1"]
    negative_tweets = [s[3].strip() for s in split_data if s[1] == "0"]

    # Split by words
    x_text = positive_tweets + negative_tweets
    x_text = [data_helpers.clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_tweets]
    negative_labels = [[1, 0] for _ in negative_tweets]
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

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

    cnn = SentimentCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters
    )

    cnn.run(x_train, y_train, x_test, y_test, batch_size, num_epochs, eval_at, keep_prob)


if __name__ == "__main__":

    tweets(
        test_ratio=0.2,
        embedding_size=128,
        filter_sizes=[3,4,5],
        num_filters=128,
        batch_size=50,
        num_epochs=200,
        eval_at=100,
        keep_prob=0.5
    )
