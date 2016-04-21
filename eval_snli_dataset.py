# -*- coding: utf-8 -*-

# Created by junfeng on 4/6/16.

# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('eval_rte_dataset')

from datetime import datetime

logger_filename = './log/eval_snli_dataset-{0}.log'.format(datetime.now())
import os.path
# import os
# if os.path.isfile(logger_filename):
#     os.remove(logger_filename)
file_handler = logging.FileHandler(logger_filename, mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise

import numpy as np
import pandas as pd

import skipthoughts


def feats(ts, hs):
    pass


def eval_kfold(A, B, train, labels, shuffle=True, k=10, seed=1234, use_feats=False):
    """
    Perform k-fold cross validation
    """
    # features
    labels = np.array(labels)
    if use_feats:
        features = np.c_[np.abs(A - B), A * B, feats(train[0], train[1])]
    else:
        features = np.c_[np.abs(A - B), A * B]

    scan = [2**t for t in range(0,9,1)]
    npts = len(features)
    kf = KFold(npts, n_folds=k, shuffle=shuffle, random_state=seed)
    scores = []

    for s in scan:

        scanscores = []

        for train, test in kf:

            # Split data
            X_train = features[train]
            y_train = labels[train]
            X_test = features[test]
            y_test = labels[test]

            # Train classifier
            clf = LogisticRegression(C=s)
            clf.fit(X_train, y_train)
            yhat = clf.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            scanscores.append(acc)
            print (s, acc)

        # Append mean score
        scores.append(np.mean(scanscores))
        print scores

    # Get the index of the best score
    s_ind = np.argmax(scores)
    s = scan[s_ind]
    print scores
    print s
    return s


def read_rte_data():
    logger.info('read data ...')
    vectorized_ts, vectorized_hs, labels = joblib.load('./data/processed-rte-dataset.pkl')
    return vectorized_ts, vectorized_hs, labels


def gen_cv():
    vectorized_ts, vectorized_hs, labels = read_rte_data()
    X_all = np.concatenate((vectorized_ts, vectorized_hs), axis=1)
    logger.info('X_all.shape: {0}'.format(X_all.shape))
    logger.info('labels.shape: {0}'.format(labels.shape))
    skf = StratifiedKFold(labels, n_folds=3, shuffle=True, random_state=919)
    return skf, X_all, labels


def run():
    mean_acc = 0.0
    mean_logloss = 0.0
    skf, X_all, labels = gen_cv()
    for fold, (test_index, train_index) in enumerate(skf, start=1):
        logger.info('at fold: {0}'.format(fold))
        logger.info('train samples: {0}, test samples: {1}'.format(len(train_index), len(test_index)))
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        rfc = RandomForestClassifier(n_jobs=10, random_state=919)
        rfc.fit(X_train, y_train)
        y_test_predicted = rfc.predict(X_test)
        y_test_proba = rfc.predict_proba(X_test)
        # equals = y_test == y_test_predicted
        # acc = np.sum(equals) / float(len(equals))
        acc = accuracy_score(y_test, y_test_predicted)
        logger.info('test data predicted accuracy: {0}'.format(acc))
        # log loss -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
        logloss = log_loss(y_test, y_test_proba)
        logger.info('log loss at test data: {0}'.format(logloss))
        # logger.info('log loss at test data using label: {0}'.format(log_loss(y_test, y_test_predicted)))
        mean_acc += acc
        mean_logloss += logloss

    n_folds = skf.n_folds
    logger.info('mean acc: {0}'.format(mean_acc / n_folds))
    logger.info('mean log loss: {0}'.format(mean_logloss / n_folds))


def get_sentence_sample(df):
    ts = []
    hs = []
    labels = []
    samples = []
    for i, row in df.iterrows():
        gold_label = row.gold_label
        if gold_label == '-':
            continue
        elif gold_label == 'entailment':
            value = 1
        else:
            value = 0
        labels.append(value)
        t = row.sentence1
        h = row.sentence2
        if type(h) != str:
            print(row)
            raise TypeError("h is not str")
        ts.append(t)
        hs.append(h)
        samples.append(u'{0} {1}'.format(t, h))
        if i % 1000 == 0:
            logger.info('processed sample {0}'.format(i))

    labels = np.array(labels)
    sample_length = len(labels)
    logger.info('sample length: {0}'.format(sample_length))
    logger.info('unique ts: {0}, unique hs: {1}'.format(len(set(ts)), len(set(hs))))
    logger.info('unique sample: {0}'.format(len(set(samples))))
    logger.info('TRUE labels: {0}'.format(np.sum(labels)))
    return ts, hs, labels


def read_model():
    model = skipthoughts.load_model()
    return model


def read_snli_from_csv(model):
    train_saved_path = './snli/processed-train.pkl'
    dev_saved_path = './snli/processed-dev.pkl'
    test_saved_path = './snli/processed-test.pkl'
    if os.path.isfile(train_saved_path) and os.path.isfile(test_saved_path):
        X_train, train_labels = joblib.load(train_saved_path)
        X_test, test_labels = joblib.load(test_saved_path)
        return X_train, X_test, train_labels, test_labels

    if model is None:
        raise ValueError("model is None")

    train_df = pd.read_csv('./snli/snli_1.0/snli_1.0_train.txt', delimiter='\t')
    train_df = train_df[pd.notnull(train_df.sentence2)]
    train_df = train_df[train_df.gold_label != '-']
    train_df = train_df[:(len(train_df) / 3)]

    train_ts, train_hs, train_labels = get_sentence_sample(train_df)
    logger.info('encoding train samples ...')
    logger.info('encoding ts ...')
    vectorized_train_ts = skipthoughts.encode(model, train_ts)
    logger.info('encoding hs ...')
    vectorized_train_hs = skipthoughts.encode(model, train_hs)
    del train_df, train_ts, train_hs
    X_train = np.concatenate((vectorized_train_ts, vectorized_train_hs), axis=1)
    logger.info('dump to file ...')
    joblib.dump((X_train, train_labels), train_saved_path)

    test_df = pd.read_csv('./snli/snli_1.0/snli_1.0_test.txt', delimiter='\t')
    test_df = test_df[pd.notnull(test_df.sentence2)]
    test_df = test_df[test_df.gold_label != '-']

    test_ts, test_hs, test_labels = get_sentence_sample(test_df)
    logger.info('encoding test samples ...')
    logger.info('encoding ts ...')
    vectorized_test_ts = skipthoughts.encode(model, test_ts)
    logger.info('encoding hs ...')
    vectorized_test_hs = skipthoughts.encode(model, test_hs)
    del test_df, test_ts, test_hs
    X_test = np.concatenate((vectorized_test_ts, vectorized_test_hs), axis=1)

    logger.info('dump to file ...')
    joblib.dump((X_test, test_labels), test_saved_path)
    logger.info('done')

    return X_train, X_test, train_labels, test_labels


def logistic_test():
    logger.info('using logistic regression')
    logger.info('read model ...')
    n_components = None
    C = 4
    # model = read_model()
    model = None

    logger.info('loading data ...')
    X_train, X_test, train_labels, test_labels = read_snli_from_csv(model)
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
    # pca = PCA(n_components=n_components)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # logger.info('After PCA')
    # logger.info('X_train.shape: {0}'.format(X_train.shape))
    # logger.info('X_test.shape: {0}'.format(X_test.shape))
    logreg = LogisticRegression(C=C, n_jobs=10, random_state=919)
    logreg.fit(X_train, train_labels)
    y_test_predicted = logreg.predict(X_test)
    y_test_proba = logreg.predict_proba(X_test)
    acc = accuracy_score(test_labels, y_test_predicted)
    logger.info('evaluate at snli test data')
    logger.info('test data predicted accuracy: {0}'.format(acc))
    # logloss = log_loss(test_labels, y_test_proba)
    # logger.info('log loss at test data: {0}'.format(logloss))


def random_forest_test():
    logger.info('read model ...')
    n_components = 256
    model = read_model()
    X_train_all = []
    X_test_all = []
    train_labels_all = []
    test_labels_all = []
    for version in range(1, 4):
        logger.info('loading version {0} data ...'.format(version))
        X_train, X_test, train_labels, test_labels = read_snli_from_csv(model)
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        train_labels_all.append(train_labels)
        test_labels_all.append(test_labels)
        logger.info('X_train.shape: {0}'.format(X_train.shape))
        logger.info('X_test.shape: {0}'.format(X_test.shape))
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        logger.info('After PCA')
        logger.info('X_train.shape: {0}'.format(X_train.shape))
        logger.info('X_test.shape: {0}'.format(X_test.shape))
        rfc = RandomForestClassifier(n_jobs=10, random_state=919)
        rfc.fit(X_train, train_labels)
        y_test_predicted = rfc.predict(X_test)
        y_test_proba = rfc.predict_proba(X_test)
        acc = accuracy_score(test_labels, y_test_predicted)
        logger.info('evaluate at RTE {0} dataset'.format(version))
        logger.info('test data predicted accuracy: {0}'.format(acc))
        logloss = log_loss(test_labels, y_test_proba)
        logger.info('log loss at test data: {0}'.format(logloss))

    X_train_all = np.concatenate(X_train_all)
    X_test_all = np.concatenate(X_test_all)
    train_labels_all = np.concatenate(train_labels_all)
    test_labels_all = np.concatenate(test_labels_all)
    logger.info('X_train_all.shape: {0}'.format(X_train_all.shape))
    logger.info('X_test_all.shape: {0}'.format(X_test_all.shape))
    pca = PCA(n_components=n_components)
    X_train_all = pca.fit_transform(X_train_all)
    X_test_all = pca.transform(X_test_all)
    logger.info('After PCA')
    logger.info('X_train_all.shape: {0}'.format(X_train_all.shape))
    logger.info('X_test_all.shape: {0}'.format(X_test_all.shape))
    rfc = RandomForestClassifier(n_jobs=10, random_state=919)
    rfc.fit(X_train_all, train_labels_all)
    y_test_all_predicted = rfc.predict(X_test_all)
    y_test_all_proba = rfc.predict_proba(X_test_all)
    acc = accuracy_score(test_labels_all, y_test_all_predicted)
    logger.info('evaluate at RTE combined dataset')
    logger.info('test data predicted accuracy: {0}'.format(acc))
    logloss = log_loss(test_labels_all, y_test_all_proba)
    logger.info('log loss at test data: {0}'.format(logloss))

if __name__ == '__main__':
    logistic_test()
