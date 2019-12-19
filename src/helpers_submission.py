#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:18:31 2019

@author: Juliane
"""
import numpy as np
import pickle
import csv
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gensim.models
from mpl_toolkits import mplot3d
import tensorflow as tf
import tempfile

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def vectorization_train_submission(embedding_dim):
    """Function converting the training dataset into its embedding encoding.
    Each tweet is converted into its token embedding matrix, and into its full sentence embedding vector.
    Outputs:
        - feat_vectors: average sentence embedding vectors of the training dataset
        - labels: output labels of the training dataset
        - feat_matrices : token embedding matrices for each tweet sample of the training dataset 
        - pca model fitted on the training dataset """
        
    w2v_embedding(embedding_dim)
    
    # Load token dictionnary
    with open('vocab_word2vec'+emdedding_dim+'_RUBY.pkl', 'rb') as f: 
        vocab = pickle.load(f)
        
    # Load word embeddings
    embeddings = np.load('embedding_word2vec'+emdedding_dim+'_RUBY.npy')
    
    embeddings, pca_model = PCA_reduction(embeddings,embedding_dim)
    
    embedding_size = embeddings.shape[1]


        
    labels = build_labels('../data/train_pos_ruby.txt', '../data/train_neg_ruby.txt')
    
    feat_vectors = []
    feat_matrices = []

    
    for fn in ['../data/train_pos_ruby.txt', '../data/train_neg_ruby.txt']: # TODO : replace by full datasets
        with open(fn) as f:
            for line in f:
                
                tokens = [vocab.get(t, -1) for t in line.strip().split()] # .split : By default any whitespace is a separator
                # returns index of corresponding token or -1 if not found
                tokens = [t for t in tokens if t >= 0] # deletes -1 values
                if tokens: # verify that at least one token
                    feat_matrix = embeddings[tokens,:]
                    feat_matrices.append(feat_matrix)
                    feat_vectors.append(np.mean(feat_matrix, axis = 0))

                else:
                    # If a tweet contains no valid token, add zero vector/matrix
                    feat_matrices.append(np.zeros((1,embedding_size)))
                    feat_vectors.append(np.mean(feat_matrix, axis = 0))

   
    # Compute length of largest tweet in terms of number of token
    MAXIMAL_TWEET_LENGTH = compute_maximal_length(np.array(feat_matrices))
    # Pad matrices to have a fixed input size
    feat_matrices = pad_matrices(feat_matrices, MAXIMAL_TWEET_LENGTH)
    
    # Check for NaN values
    print("Number of NaN values : {}".format(np.sum(np.isnan(feat_vectors)))) # = 0 --> no NaN values
    

    return feat_vectors, labels, feat_matrices, pca_model, MAXIMAL_TWEET_LENGTH


def vectorization_test_submission(pca_model, maximal_length, embedding_dim):
    """Function converting the test dataset into its embedding encoding.
    Each tweet is converted into its token embedding matrix, and into its full sentence embedding vector.
    Outputs:
        - feat_vectors: average sentence embedding vectors of the test dataset
        - feat_matrices : token embedding matrices for each tweet sample of the test dataset """
    
    w2v_embedding(embedding_dim)
    
    
    # Load token dictionnary
    with open('vocab_word2vec'+emdedding_dim+'_RUBY.pkl', 'rb') as f: 
        vocab = pickle.load(f)
        
    # Load word embeddings
    embeddings = np.load('embedding_word2vec'+emdedding_dim+'_RUBY.npy')
    
    embeddings = PCA_reduction_test(embeddings, pca_model)
    
    embedding_size = embeddings.shape[1]


     
    feat_vectors = []
    feat_matrices = []

    
    
    with open('../test_data_ruby.txt') as f:
        for line in f:
            
            tokens = [vocab.get(t, -1) for t in line.strip().split()] # .split : By default any whitespace is a separator
            # returns index of corresponding token or -1 if not found
            tokens = [t for t in tokens if t >= 0] # deletes -1 values
            if tokens: # verify that at least one token
                feat_matrix = embeddings[tokens,:] 
                feat_matrices.append(feat_matrix)
                feat_vectors.append(np.mean(feat_matrix, axis = 0))

            else:
                # If a tweet contains no valid token, add zero vector/matrix
                feat_matrix = np.zeros((1,embedding_size))
                feat_matrices.append(feat_matrix)
                feat_vectors.append(np.mean(feat_matrix, axis = 0))
 
    
    feat_matrices = pad_matrices(feat_matrices, maximal_length)
    
    # Check for NaN values
    print("Number of NaN values : {}".format(np.sum(np.isnan(feat_vectors)))) # = 0 --> no NaN values

    return feat_vectors, feat_matrices


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


def PCA_reduction(data,nb_pc):
    """ Function running a Principal Component Analysis on the feature matrix, rotating it in the new dimensional space
    and keeping only the more relevant principal components.
    Inputs:
        - data : original feature matrix
        - nb_pc : number of principal components to keep
    Outputs:
        - pc : transformed feature matrix
        - pca_model : pca model fitted to the training dataset """
        
    data = StandardScaler().fit_transform(data)
    pca_model = PCA(n_components=nb_pc)
    pc = pca_model.fit_transform(data)
    return pc, pca_model

def PCA_reduction_test(data, pca_model):
    """ Function applying the PCA model fitted on the training set to the test sets.
    Inputs:
        - data : original feature matrix
        - nb_pc : number of principal components to keep
    Outputs:
        - pc : transformed feature matrix
        - pca_model : pca model fitted to the training dataset """
    data = StandardScaler().fit_transform(data)
    pc = pca_model.transform(data)
    return pc

def compute_maximal_length(matrix_3D):
    """Compute the size of the largest tweet (in terms of number of tokens)
    Input:
        - feature_matrix
    Output:
        - length of the longest tweet"""

    max_length = 0
    for i in range(matrix_3D.shape[0]):
        if matrix_3D[i].shape[0] > max_length:
            max_length = matrix_3D[i].shape[0]
    return max_length

def pad_matrices(feat_matrices, length):
    """ Function adding zero-padding to feature matrices, in order to have fixed input size.
    Input:
        - feat_matrices: token embedding matrices for each tweet sample of the training dataset
        - length : padding target length
    Output :
        - padded feature matrices"""
    return pad_sequences(feat_matrices, maxlen=length)

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
    pickle_vocab(embedding_dim)
    
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
        