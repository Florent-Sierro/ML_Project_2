#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:43 2019

@author: Alice
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gensim
import gensim.models as models
import gensim.utils as utils
import tempfile
import random
import scipy
from scipy.linalg import svd
from scipy.sparse import coo_matrix
import itertools
import collections
from collections import Counter
import pickle
from glove import Corpus, Glove

def standardize(data):
    '''
    DESCRIPTION : Standardize columns of input array by mean substraction and std normalisation

    INPUT:
        |--- data: [array] 2D array to be standardized with lines as samples, features as columns
    OUTPUT:   
        |---data_std : [arr] 2D array standardized with zero mean and unit variance over each column
    '''
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    return data_std

def get_tweets(data):
    ''' 
    DESCRIPTION : Format tweets to perform sentence embedding using Gensim
    
    INPUT: 
        |--- data: [array] array one string by tweet  
    OUTPUT:
        |--- Gensim model object containing each tweet tokenized and associated with its index
    '''

    for i, tweet in enumerate(data):
        tokens = utils.simple_preprocess(tweet)
        yield models.doc2vec.TaggedDocument(tokens, [i])
        
def get_tokens(data):
    ''' 
    DESCRIPTION: Get array of tokens by tweet
    
    INPUT:
        |--- data: [array] array one string by tweet
    OUTPUT:
        |--- tweets: [array] array with one list of tokens by tweet
    '''
    tweets = []
    for line in data:
        tweets.append(line.strip().split())
    return tweets

def get_average(data,embeddings,vocab,embed_dim):
    '''
        DESCRIPTION: for each tweet, performs averaging of embeddings from all relevant tokens in each tweet

        INPUT : 
            |--- data: [list] list of tweets
            |--- embeddings: [array] 2D array total number of words in corpus x emebedding size, each line is a single word embedding
            |--- embed_dim: [int] scalar for number of embedding dimension
        OUTPUT:
            |--- feat_matrices: [list] list of 2D array with each array containing embedding for each word in a single tweet, third dimension is number of tweets
            |--- feat_vectors: [list] list of 1D array nb tweets x embedding size
    '''
    feat_matrices = []
    feat_vectors = []

    for line in data:
        # Get the token index list for each tweet (-1 if not found in the dictionnary)
        tokens = [vocab.get(t, -1) for t in line.strip().split()] # .split : By default any whitespace is a separator
        tokens = [t for t in tokens if t >= 0] # deletes -1 values

        if tokens: # verify that at least one token
            feat_matrix = embeddings[tokens,:] # word embedding matrix for the tweet
            feat_matrices.append(feat_matrix)
            feat_vectors.append(np.mean(feat_matrix, axis = 0)) # sentence embedding vector for the tweet (mean vector)
        else:
            # Keep tract of samples comprising zero valid token 
            feat_matrix = np.zeros((1,embed_dim))
            feat_matrices.append(feat_matrix)
            feat_vectors.append(np.mean(feat_matrix, axis = 0))

    return feat_matrices, feat_vectors

def map_word_frequency(corpus):
    '''
        DESCRIPTION: counting occurence of tokens in corpus 

        INPUT: 
            |--- corpus : [list] list of tweets
        OUTPUT: 
            |--- Counter : [dict] dictionnary with tokens as keys, number of occurrence in Corpus as value
    '''
    return Counter(itertools.chain(*corpus))

def get_Arora(data, embeddings, vocab,embedding_size, a):
    '''
	    DESCRIPTION : tweet embedding by computing weighted average of the word vectors in the tweets; 
        remove the projection of the average vectors on their first principal component.
	    Borrowed from https://github.com/peter3125/sentence2vec; now compatible with python 2.7
        Inspired by Arora et al. 2017 : A Simple but though to beat baseline for sentence embedding

        INPUT: 
            |--- tokenised_sentence_list: [list] list of lists of tokens per tweet
            |--- embedding_size: [int] size of each token vector representation 
            |--- word_emb_model: [GENSIM model] Gensim model for word embedding 
            |--- a: [float] weighting parameters
        OUTPUT:
            |--- sentence_vs: [list] list of 1D numpy array containing vector representation of single tweet
	'''
    
    tokenised_sentence_list= get_tokens(data)
    word_counts = map_word_frequency(tokenised_sentence_list)
    sentence_set = []

    for sentence in tokenised_sentence_list:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        if not sentence : print('warning')
        for word in sentence:
            a_value = a/(a + word_counts[word])
        try:
            id = vocab.get(word, -1) # .split : By default any whitespace is a separator
            if id >= 0 :vs = np.add(vs, np.multiply(a_value, embeddings[id,:]))
        except:
            pass
        np.divide(vs,sentence_length)
        sentence_set.append(vs)

    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.explained_variance_ratio_
    u = np.multiply(u, np.transpose(u))

    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u.np.append(u,0)
    
    # common component removal
    sentence_vecs = []
    for vs in sentence_set : 
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs,sub))

    return sentence_vecs

def word2vec(data, embed_dim = 10, min_count = 5, window_size = 2):
    '''
    DESCRIPTION: Perform Work2Vec using skip-gram algo for word embedding then converted to tweet embedding
    
    INPUT:
        |--- data: [array] array one string by tweet
        |--- embed_dim: [int] scalar of number of final embedded dimensions 
        |--- min_count: [int] scalar of token counts below which tokens should be discarded
        |--- window_size: [int] scalar of n-grams size considered 
        |--- average: [bool] boolean for whether sentence embedding is performed by averaging word embeddings
    OUTPUT:
        |--- embeddings: [arr] 2D numpy array with each row is the embedding of a single token 
        |--- vocab: [dict] dict with tokens as keys, index of the word embedding of this token in the embeddings matrix, as values
    ''' 
    
    tokenised_sentence_list = get_tokens(data)
    model = models.Word2Vec(tokenised_sentence_list, min_count=min_count,size=embed_dim, workers=3, window=window_size, sg = 1)

    training_loss = model.get_latest_training_loss()
    print("Train Loss {:05f}".format(training_loss))

    embeddings = np.zeros((len([*model.wv.vocab]),embed_dim))
    for i ,token in enumerate([*model.wv.vocab]):
        embeddings[i,:] = model.wv[token]

    vocab = dict()
    for idx, line in enumerate([*model.wv.vocab]):
        vocab[line.strip()] = idx
    
    return embeddings, vocab, model

def get_embeddings(model,data,embed_dim):
    '''
    DESCRIPTION : Combines results of model of tweet/paragraph embedding into a single feature vector by tweet
    
    INPUT:
        |--- model: [gensim model] Gensim model containing results of tweet embedding
        |--- data: [array] array with one string by tweet 
    OUTPUT:
        |--- embeddings: [dict] dictionnary with tweets as keys, embedding vector as values
    '''
    
    embeddings = np.zeros((len(data),embed_dim))
    tokens = get_tokens(data)
    print('Size data ',len(data))
    for i, tweet in enumerate(data):

        if i%10000 == 0: print("Processing {:0.3f} of the data".format(1-(i/len(data))))
        embeddings[i,:] = model.infer_vector(tokens[i])

    return embeddings

def doc2vec(data_cleaned, embed_dim = 10, min_count = 5, epochs = 50):
    '''
    DESCRIPTION : Perform Work2Vec using Paragraph Vector - Distributed Bag of Words algo for sentence embedding
    
    INPUT:
        |--- data_cleaned : [array] array one string by tweet
        |--- embed_dim: [int] scalar of number of final embedded dimensions 
        |--- min_count: [int] scalar of token counts below which tokens should be discarded
        |--- epochs: [int] scalar of number of epoches for NN training
    OUTPUT:
        |--- embedding : [dict] dictionnary with tweets as keys, embedding vector as values
        |--- vocab : [dict] dictionnary containing words as keys, index as values
        |--- model : [Gensim model] model of paragraph embedding performed on tweet data base
    ''' 
    
    data = get_tweets(data_cleaned)
    model = models.doc2vec.Doc2Vec(vector_size=embed_dim, min_count=min_count, epochs=epochs)
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    vocab = model.vocabulary
    
    embedding = get_embeddings(model,data_cleaned,embed_dim)

    return embedding, vocab, model

def load_pretrained_embed(filename, dim):
    '''
    DESCRIPTION: load pretrained GloVe word embeddings into dictionnary

    INPUT:
        |--- filename : [str] path name for pretrained GloVe embedding
        |--- dim : [int] integer representing embedding dimension
    OUTPUT:
        |--- vectors : [dict] dictionnary with tokens as keys and word embeddings (1D array) as values
    '''
    
    if not os.path.exists(filename):
        print(filename,'wrong filename')
        return None

    vectors= {} 
    with open(filename, "r") as file:
        for line in file:
            token = line.strip().split()
            features = [float(x) for x in token[1:]]
            if len(features) == dim: vectors[token[0]] = np.array(features)
    return vectors


def glove_embed(data, embed_dim, window_size, epochs_, step_size):
    '''
    DESCRIPTION : Perform Global Vectors for word embeddings for tokens in data set

    INPUT:
        |--- train: list of tweets
        |--- embed_size: [int] integer representing embedding dimension
        |--- window_size: [int] integer representing the size of the window of tokens considered during training for each token
        |--- epochs: [int] integer for number of epochs for Word2Vec training
        |--- step_size: [float] learning step for the SGD for Word2Vec training 

    OUTPUT:
        |--- embeddings: [dict] dictionnary with tweets as keys and 1D array of feature vector as values
        |--- vocab: [dict] dictionnary with tokens as keys and index of each token in vocab as values
        |--- glove: [Global Vectors Model] GloVe model trained on data
    '''
    sentences = get_tokens(data)

    model = Corpus()
    model.fit(sentences, window = window_size)

    glove = Glove(no_components=embed_dim, learning_rate=step_size)
    glove.fit(model.matrix, epochs=epochs_,no_threads=1, verbose=True)
    glove.add_dictionary(model.dictionary)

    embeddings = np.zeros((len([*glove.dictionary]),embed_dim))
    for w, id_ in glove.dictionary.items():
        embeddings[id_,:] = np.array([glove.word_vectors[id_]])

    vocab = dict()
    for idx, line in enumerate([*glove.dictionary]):
        vocab[line.strip()] = idx
    
    return embeddings, vocab, glove

def glove_embed_pretrained(data, embed_dim):
    '''
    DESCRIPTION: Generates pretrained GloVe embeddings and pretrained vocabulary under specific format

    INPUT:
        |--- data:
        |--- embed_size: [int] integer representing embedding dimension
    OUTPUT:
        |--- embeddings: [array] 2D array nb words in dictionnary x embedding size, contains word embeddings of each word in vocabulary
        |--- vocab: [dict] tokens as keys, index of each token in vocabulary as values
    '''

    root = os.getcwd() + '/drive/My Drive/MachineLearning_Project2/Pretrained_GloVe/'
    FILE_NAME = root+'glove.twitter.27B.' + str(embed_dim) +'d.txt'
    embeddings_dict = {}

    print("Processing file: " + FILE_NAME)
    temp = load_pretrained_embed(FILE_NAME,embed_dim)
    embeddings_dict.update(temp)

    vocab = dict()
    for idx, line in enumerate([*embeddings_dict]):
        vocab[line.strip()] = idx
    
    embeddings = np.asarray([*embeddings_dict.values()])
    return embeddings, vocab

def pickle_vocab(filename):

    vocab = dict()
    with open(filename+'.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(filename +'.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

def create_vocab_txt(vocab,filename):
    '''
    DESCRIPTION : Create txt which stores vocabulary dictionnary 

    INPUT:
        |--- vocab: [dict] dictionnary with keys as tokens and index of each token as values
        |--- filename: [str] path of text file to create
    '''
    with open(filename+'.txt',"w+") as f:
        for i, word in enumerate(vocab):
            f.write(word+'\n')
        f.close()


