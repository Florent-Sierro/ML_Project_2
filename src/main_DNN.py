#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:09:54 2019

@author: Juliane
"""

from helpers_DNN import *
from models_DNN import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


EMBEDDING_DIM = 30
BATCH_SIZE = 512
EPOCHS = 260

### Build simple DNN (LSTM, Classical-CNN, Multi-channel CNN)

# Vectorize text data
_, labels, feat_matrices = vectorization(pca=True, pca_comp=30, embedding_dim=EMBEDDING_DIM)


# Convert labels to 2 categorical variables, to be able to use categorical_crossentropy loss function
labels = to_categorical(labels)


X_train, X_test, y_train, y_test = train_test_split(feat_matrices, labels, test_size=0.2, random_state=1)

model = build_classical_CNN(EMBEDDING_DIM)
# model = build_multichannel_CNN(EMBEDDING_DIM)
# model = build_LSTM(EMBEDDING_DIM)



history = model.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X_test,y_test),
                      verbose=1)
plot_history(history)
print(model.summary())

### Build multi-embeddings CNN
"""
MAXIMAL_TWEET_LENGTH = 60
padded_sequences, embedding_matrix, labels, vocab_size = multichannel_vectorization(MAXIMAL_TWEET_LENGTH, pca=True, pca_comp=30, embedding_dim=EMBEDDING_DIM)

model = build_multichannel_embeddings(vocab_size, EMBEDDING_DIM, embedding_matrix, MAXIMAL_TWEET_LENGTH)

history = model.fit(padded_sequences, labels, validation_split=0.2, epochs=100, batch_size=200, verbose=2)

plot_history(history)
print(model.summary())
"""