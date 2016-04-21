# -*- coding: utf-8 -*-

# Created by junfeng on 4/6/16.

# logging config
import logging

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('eval_rte_dataset')

from datetime import datetime

logger_filename = './log/eval_rte_dataset-{0}.log'.format(datetime.now())
import os.path
# import os
# if os.path.isfile(logger_filename):
#     os.remove(logger_filename)
file_handler = logging.FileHandler(logger_filename, mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise

import numpy as np

import skipthoughts
import rte_utils


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


def get_sentence_sample(pairs):
    sample_length = len(pairs)
    logger.info('sample length: {0}'.format(sample_length))
    ts = []
    hs = []
    labels = np.empty(sample_length, dtype=int)
    samples = []
    for i, pair in enumerate(pairs):
        value = pair.value
        labels[i] = value
        t = pair.text
        h = pair.hyp
        ts.append(t)
        hs.append(h)
        samples.append(u'{0} {1}'.format(t, h))
        if i % 1000 == 0:
            logger.info('processed sample {0}'.format(i))
    logger.info('unique ts: {0}, unique hs: {1}'.format(len(set(ts)), len(set(hs))))
    logger.info('unique sample: {0}'.format(len(set(samples))))
    logger.info('TRUE labels: {0}'.format(np.sum(labels)))
    return ts, hs, labels


def read_model():
    model = skipthoughts.load_model()
    return model


def read_rte_from_nltk(model=None, version=3):
    train_saved_path = './data/processed-rte{0}-train.pkl'.format(version)
    test_saved_path = './data/processed-rte{0}-test.pkl'.format(version)
    if os.path.isfile(train_saved_path) and os.path.isfile(test_saved_path):
        X_train, train_labels = joblib.load(train_saved_path)
        X_test, test_labels = joblib.load(test_saved_path)
        return X_train, X_test, train_labels, test_labels

    if model is None:
        model = read_model()

    from nltk.corpus import rte
    train_xml = 'rte{0}_dev.xml'.format(version)
    test_xml = 'rte{0}_test.xml'.format(version)
    train_pairs = rte.pairs(train_xml)
    test_pairs = rte.pairs(test_xml)

    train_ts, train_hs, train_labels = get_sentence_sample(train_pairs)
    logger.info('encoding train samples ...')
    logger.info('encoding ts ...')
    vectorized_train_ts = skipthoughts.encode(model, train_ts)
    logger.info('encoding hs ...')
    vectorized_train_hs = skipthoughts.encode(model, train_hs)
    X_train = np.concatenate((vectorized_train_ts, vectorized_train_hs), axis=1)

    test_ts, test_hs, test_labels = get_sentence_sample(test_pairs)
    logger.info('encoding test samples ...')
    logger.info('encoding ts ...')
    vectorized_test_ts = skipthoughts.encode(model, test_ts)
    logger.info('encoding hs ...')
    vectorized_test_hs = skipthoughts.encode(model, test_hs)
    X_test = np.concatenate((vectorized_test_ts, vectorized_test_hs), axis=1)

    logger.info('dump to file ...')
    joblib.dump((X_train, train_labels), train_saved_path)
    joblib.dump((X_test, test_labels), test_saved_path)
    logger.info('done')

    return X_train, X_test, train_labels, test_labels


class RTE2Cosine(object):

    def __init__(self, word2vec_model_file):
        self.model_file = word2vec_model_file
        self.word2vec = None

    def calculate_cosine_features(self, rte_data, version=3):
        train_saved_path = './rte_data/cosine-rte{0}-train.pkl'.format(version)
        test_saved_path = './rte_data/cosine-rte{0}-test.pkl'.format(version)
        if os.path.isfile(train_saved_path) and os.path.isfile(test_saved_path):
            train_data = joblib.load(train_saved_path)
            test_data = joblib.load(test_saved_path)
            return train_data, test_data

        if self.word2vec is None:
            logger.info('loading pre-trained word2vec model ...')
            self.word2vec = Word2Vec.load_word2vec_format(self.model_file, binary=True)

        def handle(df):
            data_cosines = np.empty((len(df), 2))
            for index, row in df.iterrows():
                text = row.text
                hypothesis = row.hypothesis
                text = text.split()
                hypothesis = hypothesis.split()
                sims = np.zeros((len(text), len(hypothesis)))
                for i, w1 in enumerate(text):
                    for j, w2 in enumerate(hypothesis):
                        if w1 not in self.word2vec or w2 not in self.word2vec:
                            sim = 0.0
                        else:
                            sim = self.word2vec.similarity(w1, w2)
                        sims[i, j] = sim
                text_max_cosines = np.max(sims, axis=1)
                text_mean_cosine = np.mean(text_max_cosines)
                hypothesis_max_cosines = np.max(sims, axis=0)
                hypothesis_mean_cosine = np.mean(hypothesis_max_cosines)
                data_cosines[index, 0] = text_mean_cosine
                data_cosines[index, 1] = hypothesis_mean_cosine
            return data_cosines

        train_df = rte_data.train_df
        test_df = rte_data.test_df
        train_data = handle(train_df)
        test_data = handle(test_df)
        joblib.dump(train_data, train_saved_path)
        joblib.dump(test_data, test_saved_path)
        return train_data, test_data


def logistic_test_using_cosine(score_feature=True):
    logger.info('using cosine features in logistic regression')
    if score_feature:
        logger.info('also use score feature')
    Cs = [2**t for t in range(0, 10, 1)]
    Cs.extend([3**t for t in range(1, 10, 1)])
    rte2cosine = RTE2Cosine('/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin')
    X_train_all = []
    X_test_all = []
    train_labels_all = []
    test_labels_all = []
    for version in range(1, 4):
        logger.info('loading version {0} data ...'.format(version))
        rte_data = rte_utils.read_rte_from_nltk(version=version)
        X_train, X_test = rte2cosine.calculate_cosine_features(rte_data, version)

        train_labels = rte_data.train_df.label.values
        test_labels = rte_data.test_df.label.values
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        train_labels_all.append(train_labels)
        test_labels_all.append(test_labels)
        if score_feature:
            y_train_proba, y_test_proba = joblib.load('./rte_data/logistic_score_rte{0}.pkl'.format(version))
            X_train = np.concatenate([X_train, y_train_proba.reshape((-1, 1))], axis=1)
            X_test = np.concatenate([X_test, y_test_proba.reshape((-1, 1))], axis=1)
        logger.info('X_train.shape: {0}'.format(X_train.shape))
        logger.info('X_test.shape: {0}'.format(X_test.shape))

        logreg = LogisticRegressionCV(Cs=Cs, cv=3, n_jobs=10, random_state=919)
        logreg.fit(X_train, train_labels)
        logger.info('best C is {0}'.format(logreg.C_))
        y_test_predicted = logreg.predict(X_test)
        acc = accuracy_score(test_labels, y_test_predicted)
        logger.info('evaluate at RTE {0} dataset'.format(version))
        logger.info('test data predicted accuracy: {0}'.format(acc))

    X_train_all = np.concatenate(X_train_all)
    X_test_all = np.concatenate(X_test_all)
    if score_feature:
        y_train_all_proba, y_test_all_proba = joblib.load('./rte_data/logistic_score_rte_all.pkl')
        X_train_all = np.concatenate([X_train_all, y_train_all_proba.reshape((-1, 1))], axis=1)
        X_test_all = np.concatenate([X_test_all, y_test_all_proba.reshape((-1, 1))], axis=1)
    train_labels_all = np.concatenate(train_labels_all)
    test_labels_all = np.concatenate(test_labels_all)
    logger.info('X_train_all.shape: {0}'.format(X_train_all.shape))
    logger.info('X_test_all.shape: {0}'.format(X_test_all.shape))

    logreg = LogisticRegressionCV(Cs=Cs, cv=3, n_jobs=10, random_state=919, verbose=1)
    logreg.fit(X_train_all, train_labels_all)
    logger.info('best C is {0}'.format(logreg.C_))
    y_test_all_predicted = logreg.predict(X_test_all)
    acc = accuracy_score(test_labels_all, y_test_all_predicted)
    logger.info('evaluate at RTE combined dataset')
    logger.info('test data predicted accuracy: {0}'.format(acc))


def logistic_test(cosine_feature=True):
    logger.info('using logistic regression')
    if cosine_feature:
        logger.info('also use cosine feature')
    logger.info('read model ...')
    n_components = None
    Cs = [2**t for t in range(0, 10, 1)]
    Cs.extend([3**t for t in range(1, 10, 1)])
    # model = read_model()
    model = None
    rte2cosine = RTE2Cosine('/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin')
    X_train_all = []
    X_test_all = []
    train_labels_all = []
    test_labels_all = []
    for version in range(1, 4):
        logger.info('loading version {0} data ...'.format(version))
        train_cosine, test_cosine = None, None
        if cosine_feature:
            rte_data = rte_utils.read_rte_from_nltk(version=version)
            train_cosine, test_cosine = rte2cosine.calculate_cosine_features(rte_data, version)
        X_train, X_test, train_labels, test_labels = read_rte_from_nltk(model, version=version)
        vectorized_train_ts = X_train[:, :4800]
        vectorized_train_hs = X_train[:, 4800:]
        X_train = np.abs(vectorized_train_ts - vectorized_train_hs)
        X_train = np.concatenate([X_train, vectorized_train_ts * vectorized_train_hs], axis=1)
        if cosine_feature:
            X_train = np.concatenate([X_train, train_cosine], axis=1)
        # train_cosine_similarity = np.concatenate(
        #         map(pairwise.cosine_similarity, vectorized_train_ts, vectorized_train_hs)
        # )
        # X_train = np.concatenate([X_train, train_cosine_similarity], axis=1)
        vectorized_test_ts = X_test[:, :4800]
        vectorized_test_hs = X_test[:, 4800:]
        X_test = np.abs(vectorized_test_ts - vectorized_test_hs)
        X_test = np.concatenate([X_test, vectorized_test_ts * vectorized_test_hs], axis=1)
        if cosine_feature:
            X_test = np.concatenate([X_test, test_cosine], axis=1)
        # test_cosine_similarity = np.concatenate(
        #         map(pairwise.cosine_similarity, vectorized_test_ts, vectorized_test_hs)
        # )
        # X_test = np.concatenate([X_test, test_cosine_similarity], axis=1)

        X_train_all.append(X_train)
        X_test_all.append(X_test)
        train_labels_all.append(train_labels)
        test_labels_all.append(test_labels)
        logger.info('X_train.shape: {0}'.format(X_train.shape))
        logger.info('X_test.shape: {0}'.format(X_test.shape))
        # pca = PCA(n_components=n_components)
        # X_train = pca.fit_transform(X_train)
        # X_test = pca.transform(X_test)
        # logger.info('After PCA')
        # logger.info('X_train.shape: {0}'.format(X_train.shape))
        # logger.info('X_test.shape: {0}'.format(X_test.shape))
        logreg = LogisticRegressionCV(Cs=Cs, cv=3, n_jobs=10, random_state=919)
        logreg.fit(X_train, train_labels)
        logger.info('best C is {0}'.format(logreg.C_))
        y_test_predicted = logreg.predict(X_test)
        y_test_proba = logreg.predict_proba(X_test)
        acc = accuracy_score(test_labels, y_test_predicted)
        logger.info('evaluate at RTE {0} dataset'.format(version))
        logger.info('test data predicted accuracy: {0}'.format(acc))
        # logloss = log_loss(test_labels, y_test_proba)
        # logger.info('log loss at test data: {0}'.format(logloss))

        # save predicted score as another experience feature
        if not cosine_feature:
            y_train_proba = logreg.predict_proba(X_train)
            y_train_proba = y_train_proba[:, :1]
            y_test_proba = y_test_proba[:, :1]
            logger.info('save score ...')
            joblib.dump((y_train_proba, y_test_proba), './rte_data/logistic_score_rte{0}.pkl'.format(version))

    X_train_all = np.concatenate(X_train_all)
    X_test_all = np.concatenate(X_test_all)
    train_labels_all = np.concatenate(train_labels_all)
    test_labels_all = np.concatenate(test_labels_all)
    logger.info('X_train_all.shape: {0}'.format(X_train_all.shape))
    logger.info('X_test_all.shape: {0}'.format(X_test_all.shape))
    # pca = PCA(n_components=n_components)
    # X_train_all = pca.fit_transform(X_train_all)
    # X_test_all = pca.transform(X_test_all)
    # logger.info('After PCA')
    # logger.info('X_train_all.shape: {0}'.format(X_train_all.shape))
    # logger.info('X_test_all.shape: {0}'.format(X_test_all.shape))
    logreg = LogisticRegressionCV(Cs=Cs, cv=3, n_jobs=10, random_state=919)
    logreg.fit(X_train_all, train_labels_all)
    logger.info('best C is {0}'.format(logreg.C_))
    y_test_all_predicted = logreg.predict(X_test_all)
    y_test_all_proba = logreg.predict_proba(X_test_all)
    acc = accuracy_score(test_labels_all, y_test_all_predicted)
    logger.info('evaluate at RTE combined dataset')
    logger.info('test data predicted accuracy: {0}'.format(acc))
    # logloss = log_loss(test_labels_all, y_test_all_proba)
    # logger.info('log loss at test data: {0}'.format(logloss))

    # save predicted score as another experience feature
    if not cosine_feature:
        y_train_all_proba = logreg.predict_proba(X_train_all)
        logger.info('save score ...')
        y_train_all_proba = y_train_all_proba[:, :1]
        y_test_all_proba = y_test_all_proba[:, :1]
        joblib.dump((y_train_all_proba, y_test_all_proba), './rte_data/logistic_score_rte_all.pkl')


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
        X_train, X_test, train_labels, test_labels = read_rte_from_nltk(model, version=version)
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
    logistic_test(cosine_feature=True)
    # logistic_test_using_cosine(score_feature=False)