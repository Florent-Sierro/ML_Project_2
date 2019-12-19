#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:12:31 2019

@author: Juliane
"""

import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectFromModel
from keras.models import Sequential, Model
from keras.layers import Reshape, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, LSTM, SpatialDropout1D, GlobalMaxPooling1D, Input, concatenate, Activation, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import gensim.models
from mpl_toolkits import mplot3d
import tensorflow as tf
import tempfile
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from keras.layers import Flatten, Reshape





def PCA_reduction(data,nb_pc):
    """ Function running a Principal Component Analysis on the feature matrix, rotating it in the new dimensional space
    and keeping only the more relevant principal components.
    Inputs:
        - data : original feature matrix
        - nb_pc : number of principal components to keep
    Outputs:
        - pc : transformed feature matrix
    """
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=nb_pc)
    pc = pca.fit_transform(data)
    return pc


def vectorization(pca=False, pca_comp=100, emdedding_dim=100):
    """Function converting the full dataset into its embedding encoding.
    Each tweet is converted into its token embedding matrix, and into its full sentence embedding vector.
    Inputs: 
    - method : token vector embedding method
    - pca : boolean indicating whether a PCA is performed on embedding vectors.
    - pca_comp: number of pca components kept
    - embedding_dim : number of embedding features
    Outputs:
    - feat_vectors: average sentence embedding vectors
    - labels: corresponding
    - feat_matrices : token embedding matrices for each tweet sample. 
        Shape : (number of tweets X tweet length X embedding dimension) """

    
    w2v_embedding(embedding_dim)

        
    # Load token dictionnary
    with open('vocab_word2vec'+emdedding_dim+'_RUBY.pkl', 'rb') as f: 
        vocab = pickle.load(f)
        
    # Load word embeddings
    embeddings = np.load('embedding_word2vec'+emdedding_dim+'_RUBY.npy')
           
  
  
    # If requested, reduce the number of principal components
    if pca:
        embeddings = PCA_reduction(embeddings,pca_comp)
    else:
        embeddings = StandardScaler().fit_transform(embeddings)

    embedding_size = embeddings.shape[1]

    # Building output labels
    labels = build_labels('../data/train_pos_ruby.txt', '../data/train_neg_ruby.txt')
    
    feat_vectors = []
    feat_matrices = []

    # For each tweet of the dataset, break it down into tokens, and fetch corresponding embedding vector
    for fn in ['../data/train_pos_ruby.txt', '../data/train_neg_ruby.txt']:
        with open(fn) as f:
            for line in f:
                # Get the token index list for each tweet (-1 if not found in the dictionnary)
                tokens = [vocab.get(t, -1) for t in line.strip().split()] # .split : By default any whitespace is a separator
                tokens = [t for t in tokens if t >= 0] # deletes -1 values

                if tokens: # verify that at least one token
                    feat_matrix = embeddings[tokens,:] # word embedding matrix for the tweet
                    feat_matrices.append(feat_matrix)
                    feat_vectors.append(np.mean(feat_matrix, axis = 0)) # sentence embedding vector for the tweet (mean vector)
                else:
                    # Keep tract of samples comprising zero valid token 
                    feat_matrix = np.zeros((1,embedding_size))
                    feat_matrices.append(feat_matrix)
                    feat_vectors.append(np.mean(feat_matrix, axis = 0))
                  
          
    # Build the labeling vector (output ground truth)
    labels = np.zeros(sizes[0]+sizes[1])
    labels[:sizes[0]] = 1
    labels[sizes[0]:] = 0
    
    # Compute length of largest tweet in terms of number of token
    MAXIMAL_TWEET_LENGTH = compute_maximal_length(np.array(feat_matrices))
    # Pad matrices to have a fixed input size
    feat_matrices = pad_matrices(feat_matrices, MAXIMAL_TWEET_LENGTH)

    # Check for NaN values
    print("Number of NaN values : {}".format(np.sum(np.isnan(feat_vectors)))) # = 0 --> no NaN values

    return np.array(feat_vectors), labels, feat_matrices


def compute_maximal_length(feature_matrix):
    """Compute the size of the largest tweet (in terms of number of tokens)
    Input:
        - feature_matrix
    Output:
        - length of the longest tweet"""
  
    max_length = 0
    for i in range(feature_matrix.shape[0]):
        if feature_matrix[i].shape[0] > max_length:
            max_length = feature_matrix[i].shape[0]
    return max_length

def pad_matrices(feat_matrices, length):
    """ Function adding zero-padding to feature matrices, in order to have fixed input size.
    Input:
        - feat_matrices: token embedding matrices for each tweet sample of the training dataset
        - length : padding target length
    Output :
        - padded feature matrices"""
    return pad_sequences(feat_matrices, maxlen=length)


def multichannel_vectorization(max_tweet_length = 60, pca=False, pca_comp=100, embedding_dim=100):
    """ Vectorization method specifically implemented in order to be able to use Keras Embedding layers,
     with our pretrained embeddings.
    Inputs: 
    - method : token vector embedding method
    - max_tweet_length : maximal number of token in each tweet (to have fixed input size)
    - pca : boolean indicating whether a PCA is performed on embedding vectors.
    - pca_comp: number of pca components kept
    - embedding_dim : number of embedding features
    Outputs:
    - padded_sequences : padded TEXT sequences
    - embedding_matrix: token embedding matrices for each tweet sample
    - labels
    - vocab_size : number of words in our original dictionnary
    """
    
    w2v_embedding(embedding_dim)
    
    tweets = []
    

    # Load embeddings
    embeddings = np.load('embedding_word2vec'+embedding_dim+'_RUBY.npy')
    # Load token dictionnary
    with open('vocab_word2vec_'+embedding_dim+'_RUBY.pkl', 'rb') as f:
        vocab = pickle.load(f)
        

    vocab_size = len(vocab)

    # If requested, reduce the number of principal components
    if pca:
        embeddings = PCA_reduction(embeddings,pca_comp)
    else:
        embeddings = StandardScaler().fit_transform(embeddings)

  

  

    # Build a matrix tweets, comprising separate tweet texts
    for fn in ['../data/train_pos_ruby.txt', '../data/train_neg_ruby.txt']: # TODO : replace by full datasets
        with open(fn) as f:
            for line in f:
                tweets.append(line)

    #Break down each tweet into its tokens
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    # Padd each text sequence to have a fixed input size
    padded_sequences = pad_sequences(sequences, maxlen=max_tweet_length)
    # Get all words kept by the tokenizer
    word_index = tokenizer.word_index
    num_words = min(vocab_size, len(word_index) + 1)
    
    # Build the embedding matrix containing the word embedding vector associated to each token of the text
    # This method differs from the one used in vectorization() as embedding is done using words contained in 
    # the dictionnary built by tokenizer, but encoded by provided embedding vectors
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():

        if i >= vocab_size:
            continue

        embedding_id = vocab.get(word, -1)
        if embedding_id >= 0:
            embedding_matrix[i] = embeddings[embedding_id,:]

        else:
            embedding_matrix[i] = np.zeros((1,embedding_dim))

    # building output labels
    labels = build_labels('../data/train_pos_ruby.txt', '../data/train_neg_ruby.txt')

    return padded_sequences, embedding_matrix, labels, vocab_size

def build_labels(pos_file, neg_file):
    """ Function building output labels for both positive and negative text files
    Inputs: 
        - pos_file, neg_file: paths of the respective files
    Outputs:
        - labels : vector of output label for the whole dataset
    """
    sizes = []
    with open(pos_file) as f:
        sizes.append(len(f.readlines()))
    with open(neg_file) as f:
        sizes.append(len(f.readlines()))
      
    labels = np.zeros(sizes[0]+sizes[1])
    labels[:sizes[0]] = 1
    labels[sizes[0]:] = 0

    labels = to_categorical(labels)      
    return labels


def plot_history(history):
    """ Function plotting the training and validation loss and accuracy curves with respect to epochs. Note that additional
    running mean curves are plotted, requiring for the input history to comprise a model trained over number
    of epochs multiple of 20."""
    
    # Create mean
    x = np.linspace(0,len(history.history['acc'])-1,len(history.history['acc']))
    x_subsample = np.linspace(0,len(history.history['acc'])-1,len(history.history['acc'])/20)
    nb_lines = int(len(history.history['acc'])/20)
    mean_train = np.mean(np.reshape(np.copy(np.asarray(history.history['acc'])),(nb_lines,20)),axis=1)
    mean_val = np.mean(np.reshape(np.copy(np.asarray(history.history['val_acc'])),(nb_lines,20)),axis=1)

    # Accuracy plot
    fig, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(x,history.history['acc'],color='bisque',linewidth=2)
    ax.plot(x,history.history['val_acc'],color='lightskyblue',linewidth=2)
    ax.plot(x_subsample,mean_train,color = 'darkorange',linewidth = 2)
    ax.plot(x_subsample,mean_val,color = 'blue',linewidth=2)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Epoches')
    ax.legend(['Train Accuracy','Validation Accuracy','Mean Train Acc.','Mean Validation Acc.'],loc = 'lower right', ncol=4)
    plt.savefig('History_Accuracy')
    plt.show()
    
    mean_train = np.mean(np.reshape(np.copy(np.asarray(history.history['loss'])),(nb_lines,20)),axis=1)
    mean_val = np.mean(np.reshape(np.copy(np.asarray(history.history['val_loss'])),(nb_lines,20)),axis=1)
    
    # Loss plot
    fig, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(x,history.history['loss'],color='bisque',linewidth=2)
    ax.plot(x,history.history['val_loss'],color='lightskyblue',linewidth=2)
    ax.plot(x_subsample,mean_train,color = 'darkorange',linewidth = 2)
    ax.plot(x_subsample,mean_val,color = 'blue',linewidth=2)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoches')
    ax.legend(['Train Loss','Validation Loss','Mean Train Loss','Mean Validation Loss'],loc = 'lower right', ncol=4)
    plt.savefig('History_Loss')
    plt.show()
    
    

def get_sentences(data):
    ''' Return list of list with tokens by sentence'''
    sentences = []
    for line in data:
        for sentence in line.split('.'):
            sentences.append(sentence.split())
    return sentences 

def w2v_embedding(embedding_dim=100):
    """Function creating word embedding using Gensim's implementation of Word2Vec
    Input :
        - embedding_dim
    """
    
    train_pos = np.loadtxt('../data/train_pos_ruby.txt',delimiter='\n',dtype = str)
    train_neg = np.loadtxt('../data/train_neg_ruby.txt',delimiter='\n',dtype = str)
    train = np.concatenate((train_pos,train_neg))
   
    sentences = get_sentences(train)
    
    # training model 
    model = gensim.models.Word2Vec(sentences, min_count=5,size=embedding_dim,workers=3, window =2, sg = 1)
    training_loss = model.get_latest_training_loss()
    print(training_loss)
    
    
    with tempfile.NamedTemporaryFile(prefix='gensim-model-'+str(embedding_dim)+'-', delete=False) as tmp:
        temporary_filepath = tmp.name
        model.save(temporary_filepath)
        print(temporary_filepath)
        test = gensim.models.Word2Vec.load(temporary_filepath)
    
    embedding = []

    with open('vocab_cut_word2vec_'+str(embedding_dim)+'_RUBY.txt',"w+") as f:
        for i, word in enumerate(model.wv.vocab):
            embedding.append(model.wv[word])
            f.write(word+'\n')
    f.close()   
    
    np.save('embedding_word2vec'+str(embedding_dim)+'_RUBY',embedding)
    
def pickle_vocab(embedding_dim):
    """ Function creating the pickle format encoding of the token dictionnary created by w2v_embedding function.
    Input : 
        - embedding_dim """
    

    vocab = dict()

    with open('vocab_cut_word2vec_'+str(embedding_dim)+'_original.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx # .strip() simply removes spaces before and after tokens

    with open('vocab_word2vec_'+str(embedding_dim)+'_original.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
