# -*- coding: utf-8 -*-

# Created by junfeng on 4/15/16.

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)




import eval_snli_dataset
import eval_rte_dataset



# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None, n_features=4800):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, n_features),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=1200,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=600,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    # Finally, we'll add the fully-connected output layer, of 1 sigmoid units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices = None
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]



class EvalRTE(object):

    def __init__(self, num_epochs, batch_size):

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.rte_datasets = []
        logger.info('reading RTE datasets ...')
        for version in range(1, 4):
            logger.info('loading version {0} data ...'.format(version))
            X_train, X_test, train_labels, test_labels = eval_rte_dataset.read_rte_from_nltk(None, version=version)
            vectorized_train_ts = X_train[:, :4800]
            vectorized_train_hs = X_train[:, 4800:]
            X_train = np.abs(vectorized_train_ts - vectorized_train_hs)
            X_train = np.concatenate([X_train, vectorized_train_ts * vectorized_train_hs], axis=1)
            # train_cosine_similarity = np.concatenate(
            #         map(pairwise.cosine_similarity, vectorized_train_ts, vectorized_train_hs)
            # )
            # X_train = np.concatenate([X_train, train_cosine_similarity], axis=1)
            vectorized_test_ts = X_test[:, :4800]
            vectorized_test_hs = X_test[:, 4800:]
            X_test = np.abs(vectorized_test_ts - vectorized_test_hs)
            X_test = np.concatenate([X_test, vectorized_test_ts * vectorized_test_hs], axis=1)
            # test_cosine_similarity = np.concatenate(
            #         map(pairwise.cosine_similarity, vectorized_test_ts, vectorized_test_hs)
            # )
            # X_test = np.concatenate([X_test, test_cosine_similarity], axis=1)

            self.rte_datasets.append((version, X_train, train_labels, X_test, test_labels))

    def eval_rte_when_train_snli(self, val_fn, epoch):
        for version, X_train, train_labels, X_test, test_labels in self.rte_datasets:
            # And a full pass over the RTE dataset:
            logger.info('eval at RTE {0}'.format(version))
            train_err = 0.
            train_acc = 0.
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, train_labels, self.batch_size, shuffle=True):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1
            val_err = 0.
            val_acc = 0.
            val_batches = 0
            for batch in iterate_minibatches(X_test, test_labels, self.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  training accuracy:\t\t{:.2f} %".format(
                    train_acc / train_batches * 100))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            val_acc = val_acc / val_batches
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def run(model, num_epochs=100, batch_size=32):
    # Load the dataset
    logger.info("Loading data...")
    model = None

    X_train, X_test, train_labels, test_labels = eval_snli_dataset.read_snli_from_csv(model)
    eval_rte = EvalRTE(num_epochs, batch_size)
    vectorized_train_ts = X_train[:, :4800]
    vectorized_train_hs = X_train[:, 4800:]
    # C = eval_kfold(vectorized_train_ts, vectorized_train_hs, None, train_labels)
    X_train = np.abs(vectorized_train_ts - vectorized_train_hs)
    X_train = np.concatenate([X_train, vectorized_train_ts * vectorized_train_hs], axis=1)
    # train_cosine_similarity = np.concatenate(
    #         map(pairwise.cosine_similarity, vectorized_train_ts, vectorized_train_hs)
    # )
    # X_train = np.concatenate([X_train, train_cosine_similarity], axis=1)
    del vectorized_train_ts, vectorized_train_hs

    vectorized_test_ts = X_test[:, :4800]
    vectorized_test_hs = X_test[:, 4800:]
    X_test = np.abs(vectorized_test_ts - vectorized_test_hs)
    X_test = np.concatenate([X_test, vectorized_test_ts * vectorized_test_hs], axis=1)
    # test_cosine_similarity = np.concatenate(
    #         map(pairwise.cosine_similarity, vectorized_test_ts, vectorized_test_hs)
    # )
    # X_test = np.concatenate([X_test, test_cosine_similarity], axis=1)
    del vectorized_test_ts, vectorized_test_hs

    logger.info('X_train.shape: {0}'.format(X_train.shape))
    logger.info('X_test.shape: {0}'.format(X_test.shape))
    input_var = T.matrix('inputs')
    target_var = T.lvector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_mlp(input_var, n_features=X_train.shape[1])
    average = network

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our binary class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(average)
    prediction = prediction.flatten()
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(average, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(average, deterministic=True)
    test_prediction = test_prediction.flatten()
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(lasagne.objectives.binary_accuracy(test_prediction,
                                                            target_var))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    max_test_acc = 0.
    best_epoch = -1
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0.
        train_acc = 0.
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, train_labels, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            err, acc = val_fn(inputs, targets)
            train_acc += acc
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0.
        val_acc = 0.
        val_batches = 0
        for batch in iterate_minibatches(X_test, test_labels, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
                train_acc / train_batches * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        val_acc = val_acc / val_batches
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))

        if val_acc > max_test_acc:
            print('>> test accuracy improved {0}'.format(val_acc - max_test_acc))
            best_epoch = epoch
            max_test_acc = val_acc

        # eval RTE datasets
        eval_rte.eval_rte_when_train_snli(val_fn, epoch)

    # After training, we compute and print the test error:
    test_err = 0.
    test_acc = 0.
    test_batches = 0
    for batch in iterate_minibatches(X_test, test_labels, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    print("  best test accuracy:\t\t{:.2f} %".format(max_test_acc * 100))
    print("  at epoch {0}".format(best_epoch + 1))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on RTE using Lasagne.")
        print("Usage: %s [[EPOCHS]]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        np.random.seed(919)
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['epoch'] = int(sys.argv[1])
        # model = rte_on_skipthoughts.load_model()
        run(None, **kwargs)






