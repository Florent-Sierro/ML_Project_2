#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:43 2019

@author: Alice
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:43 2019

@author: Alice
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import helpers_tweets
reload(helpers_tweets)
from helpers_tweets import *
import helpers_training
reload(helpers_training)
from helpers_training import do_PCA, do_tsne
import sklearn
from sklearn.model_selection import train_test_split 
import os

# Load train and validation set 
root = '../data/'
TRAIN_FILENAME = root + 'train.csv'
TRAIN_LABELS_FILENAME = root + 'train_labels.pkl'
VAL_FILENAME  = root + 'val.csv'
VAL_LABELS_FILENAME = root + 'val_labels.pkl'

train = pd.read_csv(TRAIN_FILENAME)['Tweets'].values
val = pd.read_csv(VAL_FILENAME)['Tweets'].values

with open(TRAIN_LABELS_FILENAME,"rb") as f:
    train_labels = pickle.load(f)
with open(VAL_LABELS_FILENAME,"rb") as f:
    val_labels = pickle.load(f)

# Select 
train_, val_, labels_train, labels_val = train, val, train_labels, val_labels

# Set of hyperparameters to use
methods = ['w2v_avg'] # embedding method to choose from ('w2v_avg','doc2vec', 'w2v_arora','gloVe','gloVe_arora','gloVe_pretrained')
pca = True
tsne = False

EMBED_SIZE = [350]
N_GRAM = [6]
WINDOW_SIZE = [6]

if 'gloVe_pretrained' in methods:
    print("GloVe Pretrained")
    
    EMBED_SIZE = [50,100,200]
    for dim in EMBED_SIZE:

        # Word embeddings
        print("Embedding for dimension {:d}".format(dim))
        embeddings, vocab = glove_embed_pretrained(train_, dim)
        print("Successfully processed pretrained embedding")

        # Standardize data
        embeddings = standardize(embeddings)

        # PCA or tSNE
        if pca: do_PCA(embeddings)
        if tsne: do_tsne(embeddings)

        # Tweet embeddings
        _, train_embeddings = get_average(train_,embeddings,vocab,dim)
        _, val_embeddings = get_average(val_,embeddings,vocab,dim)

        folder = root + 'GloVepretrained'
        str_train = folder + '/train_' + str(dim)  + '.pkl'
        str_val = folder + '/val_' + str(dim)  + '.pkl'

        # Save 
        if os.path.exists(folder):
            with open(str_train, 'wb') as f:
                pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                f.close()
            with open(str_val, 'wb') as f:
                pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)
                f.close()
        else:
            try:
                os.mkdir(folder)
            except OSError:
                print ("Creation of the directory %s failed" % root)
            else:
                print ("Successfully created the directory %s " % root)
                with open(str_train, 'wb') as f:
                    pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                    f.close()
                with open(str_val, 'wb') as f:
                    pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)
                    f.close()
else:       
    for dim in EMBED_SIZE:
        for window in WINDOW_SIZE:
            if 'gloVe' in methods:
                print("GloVe")
                
                # Word Embeddings
                print("Embedding for dimension {:d}".format(dim))
                embeddings, vocab, model = glove_embed(train_, dim, window, 50, 0.05)

                # Standardize data
                embeddings = standardize(embeddings)

                # PCA or tSNE
                if pca: do_PCA(embeddings)
                if tsne: do_tsne(embeddings)

                # Tweet embedding 
                _, train_embeddings = get_average(train_,embeddings,vocab,dim)
                _, val_embeddings = get_average(val_,embeddings,vocab,dim)

                folder = root +'GloVe'
                str_train = folder + '/train_' + str(dim) + '_'+ str(window) + '.pkl'
                str_val = folder + '/val_' + str(dim) + '_'+ str(window) + '.pkl'

                # Save 
                if os.path.exists(folder):
                    with open(str_train, 'wb') as f:
                        pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                    with open(str_val, 'wb') as f:
                        pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)
                else:
                    try:
                        os.mkdir(folder)
                    except OSError:
                        print ("Creation of the directory %s failed" % root)
                    else:
                        print ("Successfully created the directory %s " % root)
                        with open(str_train, 'wb') as f:
                            pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                        with open(str_val, 'wb') as f:
                            pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)
            if 'gloVe_arora' in methods:
                print("GloVe Arora")

                # Word Embeddings
                print("Embedding for dimension {:d}".format(dim))
                embeddings, vocab, model = glove_embed(train_, dim, window, 50, 0.05)

                # Standardize data
                embeddings = standardize(embeddings)

                # PCA or tSNE
                if pca: do_PCA(embeddings)
                if tsne: do_tsne(embeddings)

                # Tweet embedding 
                train_embeddings = get_Arora(train_, embeddings, vocab,dim,1e-3)
                val_embeddings = get_Arora(val_, embeddings, vocab,dim,1e-3)

                folder = root +'GloVe_arora'
                str_train = folder + '/train_' + str(dim) + '_'+ str(window) + '.pkl'
                str_val = folder + '/val_' + str(dim) + '_'+ str(window) + '.pkl'

                # Save 
                if os.path.exists(folder):
                    with open(str_train, 'wb') as f:
                        pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                    with open(str_val, 'wb') as f:
                        pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)
                else:
                    try:
                        os.mkdir(folder)
                    except OSError:
                        print ("Creation of the directory %s failed" % root)
                    else:
                        print ("Successfully created the directory %s " % root)
                        with open(str_train, 'wb') as f:
                            pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                        with open(str_val, 'wb') as f:
                            pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)

    for dim in EMBED_SIZE:
        for gram in N_GRAM:
            print("Embedding for dimension {:d}, gram {:d}".format(dim,gram))

            if 'w2v_avg' in methods : 
                print("Average")

                # Word Embeddings
                print("Embedding for dimension {:d}".format(dim))
                embeddings, vocab, model = word2vec(train_, embed_dim = dim, min_count = 5, window_size = gram)

                # Standardize
                embeddings = standardize(embeddings)

                # PCA or tSNE
                if pca: do_PCA(embeddings)
                if tsne: do_tsne(embeddings)

                # Tweet embedding 
                _, train_embeddings = get_average(train_,embeddings,vocab,dim)
                _, val_embeddings = get_average(val_,embeddings,vocab,dim)
                
                folder= root + 'w2v_avg'
                str_train = folder+ '/train_' + str(dim) + '_'+ str(gram) + '.pkl'
                str_val = folder+ '/val_' + str(dim) + '_'+ str(gram) + '.pkl'

                # Save
                if os.path.exists(folder):
                    with open(str_train, 'wb') as f:
                        pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                    with open(str_val, 'wb') as f:
                        pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)
                else:
                    try:
                        os.mkdir(folder)
                    except OSError:
                        print ("Creation of the directory %s failed" % root)
                    else:
                        print ("Successfully created the directory %s " % root)
                        with open(str_train, 'wb') as f:
                            pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                        with open(str_val, 'wb') as f:
                            pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)

            if 'w2v_arora' in methods:
                print("Arora")

                # Word Embeddings
                print("Embedding for dimension {:d}".format(dim))
                embeddings, vocab, model = word2vec(train_, embed_dim = dim, min_count = 5, window_size = gram)

                # Standardize
                embeddings = standardize(embeddings)

                # PCA or tSNE
                if pca: do_PCA(embeddings)
                if tsne: do_tsne(embeddings)

                # Tweet embedding 
                train_embeddings = get_Arora(train_, embeddings, vocab,dim,1e-3)
                val_embeddings = get_Arora(val_, embeddings, vocab,dim,1e-3)

                folder = root +'arora'
                str_train = folder + '/train_' + str(dim) + '_'+ str(gram) + '.pkl'
                str_val = folder + '/val_' + str(dim) + '_'+ str(gram) + '.pkl'

                # Save
                if os.path.exists(folder):
                    with open(str_train, 'wb') as f:
                        pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                    with open(str_val, 'wb') as f:
                        pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)
                else:
                    try:
                        os.mkdir(folder)
                    except OSError:
                        print ("Creation of the directory %s failed" % root)
                    else:
                        print ("Successfully created the directory %s " % root)
                        with open(str_train, 'wb') as f:
                            pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
                        with open(str_val, 'wb') as f:
                            pickle.dump(val_embeddings, f, pickle.HIGHEST_PROTOCOL)

            if 'doc2vec' in methods : 
                print("Doc2Vec")

                # Tweet Embeddings
                print("Embedding for dimension {:d}".format(dim))
                embeddings_train, _, model = doc2vec(train_, embed_dim = dim, min_count = 5, epochs = 100)
                embeddings_val = get_embeddings(model,val_,dim)

                folder = root + 'doc2vec'
                str_train = folder + '/train_' + str(dim) + '_'+ str(gram) + '.pkl'
                str_val = folder + '/val_' + str(dim) + '_'+ str(gram) + '.pkl'

                # Save 
                if os.path.exists(folder):
                    with open(str_train, 'wb') as f:
                        pickle.dump(embeddings_train, f, pickle.HIGHEST_PROTOCOL)
                    with open(str_val, 'wb') as f:
                        pickle.dump(embeddings_val, f, pickle.HIGHEST_PROTOCOL)
                else:
                    try:
                        os.mkdir(folder)
                    except OSError:
                        print ("Creation of the directory %s failed" % root)
                    else:
                        print ("Successfully created the directory %s " % root)
                        with open(str_train, 'wb') as f:
                            pickle.dump(embeddings_train, f, pickle.HIGHEST_PROTOCOL)
                        with open(str_val, 'wb') as f:
                            pickle.dump(embeddings_val, f, pickle.HIGHEST_PROTOCOL)







