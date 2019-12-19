#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:10:48 2019

@author: Juliane
"""

from keras.models import Sequential, Model
from keras.layers import Reshape, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, LSTM, SpatialDropout1D, GlobalMaxPooling1D, Input, concatenate, Activation, Embedding
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from keras.layers import Flatten, Reshape


def build_classical_CNN():
    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu')) #2-grams
    model.add(Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu')) #3-grams
    #model.add(Conv1D(filters=size, kernel_size=4, padding='valid', activation='relu')) #4-grams
    #model.add(Conv1D(filters=size, kernel_size=5, padding='valid', activation='relu')) #5-grams
    #model.add(Conv1D(filters=size, kernel_size=6, padding='valid', activation='relu')) #6-grams
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
    model.add(Conv1D(50, 10, activation='relu', padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv1D(50, 10, activation='relu', padding='same'))
    model.add(GlobalAveragePooling1D())

    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    return model

def build_multichannel_CNN(MAXIMAL_TWEET_LENGTH, EMBEDDING_SIZE):
    tweet_input = Input(shape=(MAXIMAL_TWEET_LENGTH, EMBEDDING_SIZE))
    
    bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_input)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)
    trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_input)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)
    #fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_input)
    #fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
    #fivegram_branch = Conv1D(filters=100, kernel_size=5, padding='valid', activation='relu', strides=1)(tweet_input)
    #fivegram_branch = GlobalMaxPooling1D()(fivegram_branch)
    #merged = concatenate([bigram_branch, trigram_branch, fourgram_branch, fivegram_branch], axis=1)
    merged = concatenate([bigram_branch, trigram_branch], axis=1)
    
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(2)(merged)
    output = Activation('softmax')(merged)
    model = Model(inputs=[tweet_input], outputs=[output])
    model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
    return model

def build_multichannel_embeddings(vocab_size, embedding_size, embedding_matrix, MAXIMAL_TWEET_LENGTH=60):


    # CHANNEL 1
    
    embedding_layer1 = Embedding(vocab_size,embedding_size, embeddings_initializer=Constant(embedding_matrix), input_length=MAXIMAL_TWEET_LENGTH, trainable=True)
    
    
    sequence_input = Input(shape=(MAXIMAL_TWEET_LENGTH,), dtype='int32')
    
    embedded_sequences1 = embedding_layer1(sequence_input)
    x1 = Conv1D(100, 2, activation='relu')(embedded_sequences1)
    x1 = Conv1D(100, 3, activation='relu')(x1)
    #x1 = Conv1D(100, 4, activation='relu')(x1)
    #x1 = Conv1D(100, 5, activation='relu')(x1)
    x1 = MaxPooling1D()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Conv1D(50, 10, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Conv1D(50, 10, activation='relu')(x1)
    x1 = GlobalMaxPooling1D()(x1)
    x1 = Dropout(0.5)(x1)
    
    
    
    # CHANNEL 2
    
    embedding_layer2 = Embedding(vocab_size,embedding_size, embeddings_initializer=Constant(embedding_matrix), input_length=MAXIMAL_TWEET_LENGTH, trainable=False)
    
    
    
    
    embedded_sequences2 = embedding_layer2(sequence_input)
    x2 = Conv1D(100, 2, activation='relu')(embedded_sequences2)
    x2 = Conv1D(100, 3, activation='relu')(x2)
    #x2 = Conv1D(100, 4, activation='relu')(x2)
    #x2 = Conv1D(100, 5, activation='relu')(x2)
    x2 = MaxPooling1D()(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Conv1D(160, 10, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Conv1D(160, 10, activation='relu')(x2)
    x2 = GlobalMaxPooling1D()(x2)
    x2 = Dropout(0.5)(x2)

    
    merged = concatenate([x1, x2], axis=1)
    merged = Dense(2, activation='softmax')(merged)
    model = Model(sequence_input, merged)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])


    return model

def build_LSTM():
    model = Sequential()
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(420, dropout = 0.2, recurrent_dropout = 0.3))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy'])
    return model