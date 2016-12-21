import tensorflow as tf
import numpy as np
import os
import time
import datetime
import scnn_utils
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
        W_embedded = tf.Variable(tf.random_uniform([vocab_size, embedding_size], 
            -1.0, 1.0))
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
            pooled = self.max_pooling_step(h, 
                [1, sequence_length - filter_size + 1, 1, 1])
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

    def run(self, x_input, y_input, x_test, y_test, batch_size, num_epochs, 
        eval_step, keep_prob):
        """Runs the entire neural network on given input."""

        batches = scnn_utils.batch_iter(list(zip(x_input, y_input)), 
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
