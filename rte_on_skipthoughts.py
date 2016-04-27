'''
Skip-thought vectors
'''
import os
import warnings
import time
import sys

import theano
import theano.tensor as tensor
from sklearn.externals import joblib
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import pandas as pd
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

from rte_utils import read_rte_from_nltk

profile = False

#-----------------------------------------------------------------------------#
# Specify model and table locations here
#-----------------------------------------------------------------------------#
path_to_models = './data/'
path_to_tables = './data/'
#-----------------------------------------------------------------------------#

path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'
_EPSILON = 10e-8


def binary_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = tensor.nnet.sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = tensor.clip(output, _EPSILON, 1.0 - _EPSILON)
    return tensor.nnet.binary_crossentropy(output, target)


# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=False)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1 ** (i_t)
    fix2 = 1. - b2 ** (i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update


def unzip(zipped):
    """
    Pull parameters from Theano shared variables
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def prepare(df, table, worddict, options, use_eos=False):
    df.loc[:, 'text'] = preprocess(df['text'])
    df.loc[:, 'hypothesis'] = preprocess(df['hypothesis'])

    seqs_text = []
    seqs_hypothesis = []
    for cc in df['text']:
        seq_text = [table[w].reshape((-1, table[w].shape[-1])) if worddict[w] > 0 else table['UNK'].reshape((-1, table['UNK'].shape[-1])) for w in cc.split()]
        if use_eos:
            seq_text.append(table['eos'])
        seqs_text.append(seq_text)
    for cc in df['hypothesis']:
        seq_hypothesis = [table[w].reshape((-1, table[w].shape[-1])) if worddict[w] > 0 else table['UNK'].reshape((-1, table['UNK'].shape[-1])) for w in cc.split()]
        if use_eos:
            seq_hypothesis.append(table['eos'])
        seqs_hypothesis.append(seq_hypothesis)
    seqs_t = seqs_text
    seqs_h = seqs_hypothesis

    lengths_t = [len(s) for s in seqs_t]
    lengths_h = [len(s) for s in seqs_h]

    n_samples = len(seqs_t)
    maxlen_t = numpy.max(lengths_t) + 1
    maxlen_h = numpy.max(lengths_h) + 1

    text_embeddings = numpy.zeros((maxlen_t, n_samples, options['dim_word']), dtype='float32')
    hypothesis_embeddings = numpy.zeros((maxlen_h, n_samples, options['dim_word']), dtype='float32')
    text_masks = numpy.zeros((maxlen_t, n_samples)).astype('float32')
    hypothesis_masks = numpy.zeros((maxlen_h, n_samples)).astype('float32')
    for idx, [s_t, s_h] in enumerate(zip(seqs_t, seqs_h)):
        s_t = numpy.concatenate(s_t)
        s_h = numpy.concatenate(s_h)
        # print(s_t.shape, s_h.shape)
        text_embeddings[:lengths_t[idx],idx] = s_t
        text_masks[:lengths_t[idx]+1,idx] = 1.
        hypothesis_embeddings[:lengths_h[idx],idx] = s_h
        hypothesis_masks[:lengths_h[idx]+1,idx] = 1.

    labels = df['label'].values
    return text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks, labels


    #
    # if use_eos:
    #     text_embedding = numpy.zeros((len(text) + 1, 1, options['dim_word']), dtype='float32')
    #     hypothesis_embedding = numpy.zeros((len(hypothesis) + 1, 1, options['dim_word']), dtype='float32')
    #     text_mask = numpy.ones((len(text) + 1, 1))
    #     hypothesis_mask = numpy.ones((len(hypothesis) + 1, 1))
    # else:
    #     text_embedding = numpy.zeros((len(text), 1, options['dim_word']), dtype='float32')
    #     hypothesis_embedding = numpy.zeros((len(hypothesis), 1, options['dim_word']), dtype='float32')
    #     text_mask = numpy.ones((len(text), 1))
    #     hypothesis_mask = numpy.ones((len(hypothesis), 1))
    #
    # for j in range(len(text)):
    #     if worddict[text[j]] > 0:
    #         text_embedding[j, 1] = table[text[j]]
    #     else:
    #         text_embedding[j, 1] = table['UNK']
    # for j in range(len(hypothesis)):
    #     if worddict[hypothesis[j]] > 0:
    #         hypothesis_embedding[j, 1] = table[hypothesis[j]]
    #     else:
    #         hypothesis_embedding[j, 1] = table['UNK']
    #
    # if use_eos:
    #     text_embedding[-1, 1] = table['<eos>']
    #     hypothesis_embedding[-1, 1] = table['<eos>']
    # label = numpy.array([label], dtype='int64')
    #
    # return ()


def build_model(tparams, options):

    opt_ret = dict()

    trng = RandomStreams(1234)
    p = 0.5
    retain_prob = 1. - p
    print('dropout: {0}'.format(p))

    # description string: #words x #samples
    # text: text sentence
    # hypothesis: hypothesis sentence
    text_embedding = tensor.tensor3('text_embedding', dtype='float32')
    # text = tensor.matrix('text', dtype='int64')
    text_mask = tensor.matrix('text_mask', dtype='float32')
    hypothesis_embedding = tensor.tensor3('hypothesis_embedding', dtype='float32')
    # hypothesis = tensor.matrix('hypothesis', dtype='int64')
    hypothesis_mask = tensor.matrix('hypothesis_mask', dtype='float32')

    label = tensor.vector('label', dtype='int64')

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, text_embedding, None, options,
                                            prefix='encoder',
                                            mask=text_mask)
    ctx = proj[0][-1]
    dec_ctx = ctx
    # dropout
    dec_ctx_dropped = dec_ctx
    dec_ctx_dropped *= trng.binomial(dec_ctx_dropped.shape, p=retain_prob, dtype=dec_ctx_dropped.dtype)
    dec_ctx_dropped /= retain_prob

    # decoder (hypothesis)
    proj_hypo = get_layer(options['decoder'])[1](tparams, hypothesis_embedding, dec_ctx, options,
                                             prefix='h_decode_t',
                                             mask=hypothesis_mask)
    proj_hypo_dropped = get_layer(options['decoder'])[1](tparams, hypothesis_embedding, dec_ctx_dropped, options,
                                             prefix='h_decode_t',
                                             mask=hypothesis_mask)
    hypo_ctx = proj_hypo[0][-1]
    hypo_ctx_dropped = proj_hypo_dropped[0][-1]
    # dropout
    hypo_ctx_dropped *= trng.binomial(hypo_ctx_dropped.shape, p=retain_prob, dtype=hypo_ctx_dropped.dtype)
    hypo_ctx_dropped /= retain_prob


    # cost (cross entropy)

    logit = get_layer('ff')[1](tparams, hypo_ctx, options, prefix='ff_logit', activ='tensor.nnet.sigmoid')
    logit_dropped = get_layer('ff')[1](tparams, hypo_ctx_dropped, options, prefix='ff_logit', activ='tensor.nnet.sigmoid')

    # flatten logit
    logit = logit.flatten()
    logit_dropped = logit_dropped.flatten()
    cost = binary_crossentropy(logit_dropped, label)
    cost = tensor.mean(cost)
    acc = tensor.mean(tensor.eq(tensor.round(logit), label))

    return text_embedding, text_mask, hypothesis_embedding, hypothesis_mask, label, cost, acc


def build_rte_model(rte_data):
    """
    Build the model on saved tables
    """


    # Load model options
    print 'Loading uni-skip model parameters...'
    with open('%s.pkl' % path_to_umodel, 'rb') as f:
        uoptions = pkl.load(f)


    # Load parameters
    # fix decoder KeyError
    uoptions['decoder'] = 'gru'
    uparams = init_params(uoptions)
    del uparams['Wemb']
    uparams = load_params(path_to_umodel, uparams)
    utparams = init_tparams(uparams)

    text_embedding, text_mask, hypothesis_embedding, hypothesis_mask, label, cost, acc = build_model(utparams, uoptions)
    inps = [text_embedding, text_mask, hypothesis_embedding, hypothesis_mask, label]

    # before any regularizer
    print 'Building f_acc...',
    f_acc = theano.function(inps, acc, profile=False)
    print 'Done'

    # weight decay, if applicable
    decay_c = 0.0
    if decay_c > 0.:
        print('weight decay: {0}'.format(decay_c))
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in utparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Done'
    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=list(utparams.itervalues()))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in utparams.iteritems()], profile=False)

    grad_clip=5.
    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    optimizer = 'adam'
    f_grad_shared, f_update = eval(optimizer)(lr, utparams, grads, inps, cost)

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    batch_size = 32
    maxlen_w = 30
    max_epochs = 20
    n_words = 20000
    dispFreq = 1
    saveFreq = 50
    saveto = './rte_toy.npz'

    words = []
    print('load utable ...')
    utable = numpy.load(path_to_tables + 'utable.npy')
    f = open(path_to_tables + 'dictionary.txt', 'rb')
    for line in f:
        words.append(line.decode('utf-8').strip())
    f.close()
    utable = OrderedDict(zip(words, utable))
    # word dictionary and init
    worddict = defaultdict(lambda : 0)
    for w in utable.keys():
        worddict[w] = 1

    use_eos=False

    (text_embeddings, text_masks,
     hypothesis_embeddings, hypothesis_masks,
     labels) = prepare(rte_data.train_df[:32], utable, worddict, uoptions, use_eos)


    uidx = 0
    lrate = 0.01
    for eidx in xrange(max_epochs):
        n_samples = 0
        print 'Epoch ', eidx
        train_df = rte_data.train_df
        train_df = train_df.reindex(numpy.random.permutation(train_df.index))
        epoch_start = time.time()
        for start_i in range(0, len(train_df), batch_size):
            batched_df = train_df[start_i:start_i+batch_size]
            (
                text_embeddings, text_masks,
                hypothesis_embeddings, hypothesis_masks,
                labels) = prepare(batched_df, utable, worddict, uoptions, use_eos)

            n_samples += len(batched_df)
            uidx += 1

            ud_start = time.time()
            cost = f_grad_shared(text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks, labels)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                params = unzip(utparams)
                numpy.savez(saveto, history_errs=[], **params)
                pkl.dump(uoptions, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

        print 'Seen %d samples' % n_samples

        print('eval at train data ...')
        (
            text_embeddings, text_masks,
            hypothesis_embeddings, hypothesis_masks,
            labels) = prepare(rte_data.train_df, utable, worddict, uoptions, use_eos)
        train_acc = f_acc(text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks, labels)
        print('train acc: {0}'.format(train_acc))

        print('evaluate at test data ...')
        (
            text_embeddings, text_masks,
            hypothesis_embeddings, hypothesis_masks,
            labels) = prepare(rte_data.test_df, utable, worddict, uoptions, use_eos)
        test_acc = f_acc(text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks, labels)
        print('test acc: {0}'.format(test_acc))
        print('Epoch: {0}, used: {1} seconds'.format(eidx, time.time() - epoch_start))




    # TODO
    # print('Loading uni-skip model parameters...')
    # with open('%s.pkl' % path_to_bmodel, 'rb') as f:
    #     boptions = pkl.load(f)
    # bparams = init_params_bi(boptions)
    # bparams = load_params(path_to_bmodel, bparams)
    # btparams = init_tparams(bparams)
    #
    # # Extractor functions
    # print 'Compiling encoders...'
    # embedding, x_mask, ctxw2v = build_encoder(utparams, uoptions)
    # f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')
    #
    # embedding, x_mask, ctxw2v = build_encoder_bi(btparams, boptions)
    # f_w2v2 = theano.function([embedding, x_mask], ctxw2v, name='f_w2v2')
    #
    # # Tables
    # print 'Loading tables...'
    # utable, btable = load_tables()
    #
    # # Store everything we need in a dictionary
    # print 'Packing up...'
    # model = {}
    # model['uoptions'] = uoptions
    # model['boptions'] = boptions
    # model['utable'] = utable
    # model['btable'] = btable
    # model['f_w2v'] = f_w2v
    # model['f_w2v2'] = f_w2v2
    #
    # return model


def load_model():
    """
    Load the model with saved tables
    """
    # Load model options
    print 'Loading model parameters...'
    with open('%s.pkl'%path_to_umodel, 'rb') as f:
        uoptions = pkl.load(f)
    with open('%s.pkl'%path_to_bmodel, 'rb') as f:
        boptions = pkl.load(f)

    # Load parameters
    # fix decoder KeyError
    uoptions['decoder'] = 'gru'
    uparams = init_params(uoptions)
    uparams = load_params(path_to_umodel, uparams)
    utparams = init_tparams(uparams)

    boptions['decoder'] = 'gru'
    bparams = init_params_bi(boptions)
    bparams = load_params(path_to_bmodel, bparams)
    btparams = init_tparams(bparams)

    # Extractor functions
    print 'Compiling decoders ...'
    text_embedding, text_mask, hypothesis_embedding, hypothesis_mask, hypo_ctx = build_decoder(utparams, uoptions)
    f_w2v = theano.function([text_embedding, text_mask, hypothesis_embedding, hypothesis_mask,], hypo_ctx, name='f_w2v')

    text_embedding, text_mask, hypothesis_embedding, hypothesis_mask, hypo_ctx = build_decoder_bi(btparams, boptions)
    f_w2v2 = theano.function([text_embedding, text_mask, hypothesis_embedding, hypothesis_mask], hypo_ctx, name='f_w2v2')

    # Tables
    print 'Loading tables...'
    utable, btable = load_tables()

    # Store everything we need in a dictionary
    print 'Packing up...'
    model = {}
    model['uoptions'] = uoptions
    model['boptions'] = boptions
    model['utable'] = utable
    model['btable'] = btable
    model['f_w2v'] = f_w2v
    model['f_w2v2'] = f_w2v2

    return model


def load_tables():
    """
    Load the tables
    """
    words = []
    utable = numpy.load(path_to_tables + 'utable.npy')
    btable = numpy.load(path_to_tables + 'btable.npy')
    f = open(path_to_tables + 'dictionary.txt', 'rb')
    for line in f:
        words.append(line.decode('utf-8').strip())
    f.close()
    utable = OrderedDict(zip(words, utable))
    btable = OrderedDict(zip(words, btable))
    return utable, btable


def decode(model, rte_data, use_norm=True, verbose=True, batch_size=32, use_eos=False):

    # word dictionary and init
    numpy.random.seed(919)
    worddict = defaultdict(lambda : 0)
    for w in model['utable'].keys():
        worddict[w] = 1
    train_df = rte_data.train_df
    test_df = rte_data.test_df

    def batched_decode(df):
        s = time.time()
        data = []
        r_data = []
        df = df.reindex(numpy.random.permutation(df.index))
        for start_i in range(0, len(df), batch_size):
            if verbose:
                print(start_i)
            batched_df = df[start_i:start_i+batch_size]
            text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks, labels = \
                prepare(batched_df, model['utable'], worddict, model['uoptions'], use_eos)
            uff = model['f_w2v'](text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks)
            r_uff = model['f_w2v'](hypothesis_embeddings, hypothesis_masks, text_embeddings, text_masks)

            text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks, labels = \
                prepare(batched_df, model['btable'], worddict, model['boptions'], use_eos)
            bff = model['f_w2v2'](text_embeddings, text_masks, hypothesis_embeddings, hypothesis_masks)
            r_bff = model['f_w2v2'](hypothesis_embeddings, hypothesis_masks, text_embeddings, text_masks)
            if use_norm:
                for j in range(len(uff)):
                    uff[j] /= norm(uff[j])
                    bff[j] /= norm(bff[j])
                    r_uff[j] /= norm(r_uff[j])
                    r_bff[j] /= norm(r_bff[j])
            ff = numpy.concatenate([uff, bff], axis=1)
            r_ff = numpy.concatenate([r_uff, r_bff], axis=1)
            data.append(ff)
            r_data.append(r_ff)
        data = numpy.concatenate(data)
        r_data = numpy.concatenate(r_data)
        print('used {0} seconds'.format(time.time() - s))
        return data, r_data, df.label.values

    train_data, train_r_data, train_labels = batched_decode(train_df)
    test_data, test_r_data, test_labels = batched_decode(test_df)
    return train_data, train_r_data, train_labels, test_data, test_r_data, test_labels


def decode_rte_data(model, version):
    train_saved_path = './rte_data/decoded-rte{0}-train.pkl'.format(version)
    test_saved_path = './rte_data/decoded-rte{0}-test.pkl'.format(version)
    if os.path.isfile(train_saved_path) and os.path.isfile(test_saved_path):
        print('load from saved files ...')
        train_data, train_r_data, train_labels = joblib.load(train_saved_path)
        test_data, test_r_data, test_labels = joblib.load(test_saved_path)
        return train_data, train_r_data, train_labels, test_data, test_r_data, test_labels

    rte_data = read_rte_from_nltk(version)
    train_data, train_r_data, train_labels, test_data, test_r_data, test_labels = decode(model, rte_data)
    joblib.dump((train_data, train_r_data, train_labels), train_saved_path)
    joblib.dump((test_data, test_r_data, test_labels), test_saved_path)
    return train_data, train_r_data, train_labels, test_data, test_r_data, test_labels


def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X


def nn(model, text, vectors, query, k=5):
    """
    Return the nearest neighbour sentences to query
    text: list of sentences
    vectors: the corresponding representations for text
    query: a string to search
    """
    qf = decode(model, [query])
    qf /= norm(qf)
    scores = numpy.dot(qf, vectors.T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [text[a] for a in sorted_args[:k]]
    print 'QUERY: ' + query
    print 'NEAREST: '
    for i, s in enumerate(sentences):
        print s, sorted_args[i]


def word_features(table):
    """
    Extract word features into a normalized matrix
    """
    features = numpy.zeros((len(table), 620), dtype='float32')
    keys = table.keys()
    for i in range(len(table)):
        f = table[keys[i]]
        features[i] = f / norm(f)
    return features


def nn_words(table, wordvecs, query, k=10):
    """
    Get the nearest neighbour words
    """
    keys = table.keys()
    qf = table[query]
    scores = numpy.dot(qf, wordvecs.T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    words = [keys[a] for a in sorted_args[:k]]
    print 'QUERY: ' + query
    print 'NEAREST: '
    for i, w in enumerate(words):
        print w


def _p(pp, name):
    """
    make prefix-appended name
    """
    return '%s_%s'%(pp, name)


def init_tparams(params):
    """
    initialize Theano shared variables according to the initial parameters
    """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def load_params(path, params):
    """
    load parameters
    """
    pp = numpy.load(path)
    # not override ff_logit_W, ff_logit_b
    ff_logit_params = ['ff_logit_W', 'ff_logit_b']

    for kk, vv in params.iteritems():
        if kk in ff_logit_params:
            print('skip ff_logit layer param: {0}'.format(kk))
            continue
        if kk not in pp:
            print('{0} is not in the archive'.format(kk))
            warnings.warn('%s is not in the archive' % kk)
            continue
        print('override param {0} from archive {1}'.format(kk, path))
        params[kk] = pp[kk]
    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {
    'ff': ('param_init_fflayer', 'fflayer'),
    'gru': ('param_init_gru', 'gru_layer')
}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def init_params(options):
    """
    initialize all parameters needed for the encoder
    """
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    # encoder: GRU
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])

    # Decoder: next sentence
    # not use pre-trained decode weights
    params = get_layer(options['decoder'])[0](options, params, prefix='h_decode_t',
                                              nin=options['dim_word'], dim=options['dim'])

    # Output layer
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim'], nout=1)
    return params


def init_params_bi(options):
    """
    initialize all paramters needed for bidirectional encoder
    """
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    # encoder: GRU
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder_r',
                                              nin=options['dim_word'], dim=options['dim'])

    # Decoder: next sentence
    # not use pre-trained decode weights
    params = get_layer(options['decoder'])[0](options, params, prefix='h_decode_t',
                                              nin=options['dim_word'], dim=options['dim'])

    # Output layer
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim'], nout=1)
    return params


def build_decoder(tparams, options):
    """
    build an encoder, given pre-computed word embeddings
    """

    # description string: #words x #samples
    # text: text sentence
    # hypothesis: hypothesis sentence
    text_embedding = tensor.tensor3('text_embedding', dtype='float32')
    # text = tensor.matrix('text', dtype='int64')
    text_mask = tensor.matrix('text_mask', dtype='float32')
    hypothesis_embedding = tensor.tensor3('hypothesis_embedding', dtype='float32')
    # hypothesis = tensor.matrix('hypothesis', dtype='int64')
    hypothesis_mask = tensor.matrix('hypothesis_mask', dtype='float32')

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, text_embedding, None, options,
                                            prefix='encoder',
                                            mask=text_mask)
    ctx = proj[0][-1]
    dec_ctx = ctx

    # decoder (hypothesis)
    proj_hypo = get_layer(options['decoder'])[1](tparams, hypothesis_embedding, dec_ctx, options,
                                             prefix='decoder_f',
                                             mask=hypothesis_mask)

    hypo_ctx = proj_hypo[0][-1]

    return text_embedding, text_mask, hypothesis_embedding, hypothesis_mask, hypo_ctx


def build_decoder_bi(tparams, options):
    """
    build bidirectional encoder, given pre-computed word embeddings
    """
    # word embedding (source)
    text_embedding = tensor.tensor3('text_embedding', dtype='float32')
    text_embeddingr = text_embedding[::-1]
    text_mask = tensor.matrix('text_mask', dtype='float32')
    textr_mask = text_mask[::-1]

    hypothesis_embedding = tensor.tensor3('hypothesis_embedding', dtype='float32')
    # hypothesis = tensor.matrix('hypothesis', dtype='int64')
    hypothesis_mask = tensor.matrix('hypothesis_mask', dtype='float32')

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, text_embedding, None, options,
                                            prefix='encoder',
                                            mask=text_mask)
    projr = get_layer(options['encoder'])[1](tparams, text_embeddingr, None, options,
                                             prefix='encoder_r',
                                             mask=textr_mask)

    # ctx = tensor.concatenate([proj[0][-1], projr[0][-1]], axis=1)
    #
    # dec_ctx = ctx
    ctx = proj[0][-1]
    ctx_r = projr[0][-1]

    # decoder (hypothesis)
    proj_hypo = get_layer(options['decoder'])[1](tparams, hypothesis_embedding, ctx, options,
                                             prefix='decoder_f',
                                             mask=hypothesis_mask)
    projr_hypo = get_layer(options['decoder'])[1](tparams, hypothesis_embedding, ctx_r, options,
                                             prefix='decoder_f',
                                             mask=hypothesis_mask)

    hypo_ctx = tensor.concatenate([proj_hypo[0][-1], projr_hypo[0][-1]], axis=1)

    return text_embedding, text_mask, hypothesis_embedding, hypothesis_mask, hypo_ctx



# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin,nout=None, scale=0.1, ortho=True):
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')


# Feedforward layer
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    """
    Affine transformation + point-wise nonlinearity
    """
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    """
    Feedforward pass
    """
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])


def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    parameter init for GRU
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def gru_layer(tparams, state_below, init_state, options, prefix='gru', mask=None, **kwargs):
    """
    Feedforward pass through GRU
    """
    # nsteps is n_timesteps
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        # n_samples is 1 ?
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # the size of the hidden state
    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # W_{rz} \cdot x
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    # W \cdot x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        """

        :param m_: mask
        :param x_: state_below_
        :param xx_: state_belowx
        :param h_: previous hidden state
        :param U: horizontal stacked weight U
        :param Ux: U weight for reset gate
        :return: current hidden state
        """

        # U_{rz} \cdot h^{t-1}
        preact = tensor.dot(h_, U)
        # add
        preact += x_

        # r is reset gate
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        # u is forget gate
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # U \cdot (r \odot h^{t-1})
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        # add
        preactx = preactx + xx_

        # h is the proposed state update
        h = tensor.tanh(preactx)

        # get current step hidden state
        h = u * h_ + (1. - u) * h

        # m_[:, None] is same as m_[:, numpy.newaxis]
        # to create an axis of length one
        # apply mask to current hidden state
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state],
                                non_sequences=[tparams[_p(prefix, 'U')],
                                               tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('run with {0} version'.format(sys.argv[0]))
        sys.exit(1)
    version = int(sys.argv[1])
    print('on RTE {0}'.format(version))
    rte_data = read_rte_from_nltk(version=version)

    build_rte_model(rte_data)




