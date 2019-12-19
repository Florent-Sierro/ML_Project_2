#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:43 2019

@author: Alice
"""

import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import helpers_training
reload(helpers_training)
from helpers_training import *
import os

# Train and validation set & labels path
root = '../data/'
TRAIN_FILENAME =  root + 'train.csv'
TRAIN_LABELS_FILENAME = root+'train_labels.pkl'
VAL_FILENAME  = root+'val.csv'
VAL_LABELS_FILENAME = root + 'val_labels.pkl'

# Load train and validation set
train = pd.read_csv(TRAIN_FILENAME)['Tweets'].values
val = pd.read_csv(VAL_FILENAME)['Tweets'].values

with open(TRAIN_LABELS_FILENAME,"rb") as f:
    train_labels = pickle.load(f)
with open(VAL_LABELS_FILENAME,"rb") as f:
    val_labels = pickle.load(f)

print('Size ',train.shape)

# Set of hyperparameters to use
EMBED_SIZE = [350]
N_GRAM = [6]
WINDOW_SIZE = [6]

METHODS = ['w2v_avg'] # embedding method to choose from ('w2v_avg','doc2vec', 'w2v_arora','gloVe','gloVe_arora','gloVe_pretrained')

for dim in EMBED_SIZE:
    for window in WINDOW_SIZE:
        if 'gloVe' in METHODS:
            print("GloVe")

            # Load Embeddings
            folder= root + 'w2v_avg_file'
            TRAIN_FILENAME = folder + '/train_' + str(dim) + '_'+str(window) +'.pkl'
            VAL_FILENAME = folder + '/val_' + str(dim) + '_'+str(window) +'.pkl'

            with open(TRAIN_FILENAME,'rb') as f:
                train_features = np.array(pickle.load(f))
                f.close()
            with open(VAL_FILENAME,'rb') as f:
                val_features = np.array(pickle.load(f))
                f.close()

            # Standardize
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)

            # Perform classification with & w/0 feature selection
            df_nofeatselect, df_featselect = training_pipeline(train_features,val_features,train_labels,val_labels)

            str_1 = folder + '/results_nofeatselect' + str(dim) + '_'+ str(window) + '.csv'
            str_2 = folder + '/results_featselect' + str(dim) + '_'+ str(window) + '.csv'

            # Save 
            if os.path.exists(folder):
                df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #    df_featselect.to_csv(str_2,index = True, header = True,sep = ',')
            else:
                try:
                    os.mkdir(folder)
                except OSError:
                    print ("Creation of the directory %s failed" % root)
                else:
                    print ("Successfully created the directory %s " % root)
                    df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #        df_featselect.to_csv(str_2,index = True, header = True,sep = ',')

        if 'gloVe_arora' in METHODS:
            print("GloVe Arora")

            # Load Embeddings
            folder = root+'GloVe_arora'
            TRAIN_FILENAME = folder + '/train_' + str(dim) + '_'+str(window) +'.pkl'
            VAL_FILENAME = folder + '/val_' + str(dim) + '_'+str(window) +'.pkl'

            with open(TRAIN_FILENAME,'rb') as f:
                train_features = np.array(pickle.load(f))
                f.close()
            with open(VAL_FILENAME,'rb') as f:
                val_features = np.array(pickle.load(f))
                f.close()

            # Standardize
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)

            # Perform classification with & w/0 feature selection
            df_nofeatselect, df_featselect = training_pipeline(train_features,val_features,train_labels,val_labels)

            str_1 = folder + '/results_nofeatselect' + str(dim) + '_'+ str(window) + '.csv'
            str_2 = folder + '/results_featselect' + str(dim) + '_'+ str(window) + '.csv'

            # Save 
            if os.path.exists(folder):
                df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
                #df_featselect.to_csv(str_2,index = True, header = True,sep = ',')
            else:
                try:
                    os.mkdir(folder)
                except OSError:
                    print ("Creation of the directory %s failed" % root)
                else:
                    print ("Successfully created the directory %s " % root)
                    df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
                   #df_featselect.to_csv(str_2,index = True, header = True,sep = ',')

for dim in EMBED_SIZE:
    for gram in N_GRAM:
        print("Embedding for dimension {:d}, gram {:d}".format(dim,gram))

        if 'w2v_avg' in METHODS : 

            #Load Embeddings
            folder = root+'w2v_avg'
            TRAIN_FILENAME = folder + '/train_' + str(dim) + '_'+str(gram) +'.pkl'
            VAL_FILENAME = folder + '/val_' + str(dim) + '_'+str(gram) +'.pkl'

            with open(TRAIN_FILENAME,'rb') as f:
                train_features = np.array(pickle.load(f))
                f.close()
            with open(VAL_FILENAME,'rb') as f:
                val_features = np.array(pickle.load(f))
                f.close()

            # Standardize
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)

            # Perform classification with & w/0 feature selection
            df_nofeatselect, df_featselect = training_pipeline(train_features,val_features,train_labels,val_labels)

            str_1 = folder + '/results_nofeatselect' + str(dim) + '_'+ str(gram) + '.csv'
            str_2 = folder + '/results_featselect' + str(dim) + '_'+ str(gram) + '.csv'

            # Save 
            if os.path.exists(folder):
                df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #    df_featselect.to_csv(str_2,index = True, header = True,sep = ',')
            else:
                try:
                    os.mkdir(folder)
                except OSError:
                    print ("Creation of the directory %s failed" % root)
                else:
                    print ("Successfully created the directory %s " % root)
                    df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #        df_featselect.to_csv(str_2,index = True, header = True,sep = ',')

        if 'w2v_arora' in METHODS:
            print("W2V Arora")

            # Load Embeddings
            folder = root + 'arora'
            TRAIN_FILENAME = folder + '/train_' + str(dim) + '_'+str(gram) +'.pkl'
            VAL_FILENAME = folder + '/val_' + str(dim) + '_'+str(gram) +'.pkl'

            with open(TRAIN_FILENAME,'rb') as f:
                train_features = np.array(pickle.load(f))
                f.close()
            with open(VAL_FILENAME,'rb') as f:
                val_features = np.array(pickle.load(f))
                f.close()

            # Standardize
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)

            # Perform Classification
            df_nofeatselect, df_featselect = training_pipeline(train_features,val_features,train_labels,val_labels)

            str_1 = folder + '/results_nofeatselect' + str(dim) + '_'+ str(gram) + '.csv'
            str_2 = folder + '/results_featselect' + str(dim) + '_'+ str(gram) + '.csv'

            # Save 
            if os.path.exists(folder):
                df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #    df_featselect.to_csv(str_2,index = True, header = True,sep = ',')
            else:
                try:
                    os.mkdir(folder)
                except OSError:
                    print ("Creation of the directory %s failed" % root)
                else:
                    print ("Successfully created the directory %s " % root)
                    df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #        df_featselect.to_csv(str_2,index = True, header = True,sep = ',')
            

        if 'doc2vec' in METHODS: 
            print('doc2vec')

            # Load Embeddings
            folder = root+'doc2vec'
            TRAIN_FILENAME = folder + '/train_' + str(dim) + '_'+str(gram) +'.pkl'
            VAL_FILENAME = folder + '/val_' + str(dim) + '_'+str(gram) +'.pkl'

            with open(TRAIN_FILENAME,'rb') as f:
                train_features = np.array(pickle.load(f))
                f.close()
            with open(VAL_FILENAME,'rb') as f:
                val_features = np.array(pickle.load(f))
                f.close()

            # Standardize
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)
            # PCA 
            train_features, val_features = do_PCA(train_features, val_features, nb_pc = train_features.shape[1])
            # Perform classification with & w/0 feature selection
            df_nofeatselect, df_featselect = training_pipeline(train_features,val_features,train_labels,val_labels)

            str_1 = folder + '/results_nofeatselect' + str(dim) + '_'+ str(gram) + '.csv'
            str_2 = folder + '/results_featselect' + str(dim) + '_'+ str(gram) + '.csv'

            # Save 
            if os.path.exists(folder):
                df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #    df_featselect.to_csv(str_2,index = True, header = True,sep = ',')
            else:
                try:
                    os.mkdir(folder)
                except OSError:
                    print ("Creation of the directory %s failed" % root)
                else:
                    print ("Successfully created the directory %s " % root)
                    df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
            #        df_featselect.to_csv(str_2,index = True, header = True,sep = ',')

if 'gloVe_pretrained'  in METHODS:
    EMBED_SIZE = [50,100,200]

    for dim in EMBED_SIZE:

        #Load Embeddings
        folder = root+'GloVepretrained'
        TRAIN_FILENAME = folder + '/train_' + str(dim) +'.pkl'
        VAL_FILENAME = folder + '/val_' + str(dim) +'.pkl'

        with open(TRAIN_FILENAME,'rb') as f:
            train_features = np.array(pickle.load(f))
            f.close()
        with open(VAL_FILENAME,'rb') as f:
            val_features = np.array(pickle.load(f))
            f.close()

        #Standardize
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        # PCA
        train_features, val_features = do_PCA(train_features, val_features, nb_pc = train_features.shape[1])
        # Perform classification with & w/0 feature selection
        df_nofeatselect, df_featselect = training_pipeline(train_features,val_features,train_labels,val_labels)

        str_1 = folder + '/results_nofeatselect' + str(dim) + '.csv'
        str_2 = folder + '/results_featselect' + str(dim)  + '.csv'

        # Save 
        if os.path.exists(folder):
            df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
        #    df_featselect.to_csv(str_2,index = True, header = True,sep = ',')
        else:
            try:
                os.mkdir(folder)
            except OSError:
                print ("Creation of the directory %s failed" % root)
            else:
                print ("Successfully created the directory %s " % root)
                df_nofeatselect.to_csv(str_1,index = True, header = True,sep = ',')
        #        df_featselect.to_csv(str_2,index = True, header = True,sep = ',')





