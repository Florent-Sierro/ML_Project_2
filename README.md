### CS-433: Machine learning - Project 2 ReadMe - Twitter : A Sentiment Analysis
### Team: "BuzzLastyear" 

## Project 
In this project, we conducted a so-called Sentiment Analysis by performing a binary classification task on tweets labelled as positive or negative. 

At our disposal, a 2.5 M tweets data base with balanced classes. Smiley placed as the end of each tweet were initially used to label each tweet and were then removed.

Various Preprocessing techniques, Word Embeddings (GloVe, Word2Vec, Paragraph Embeddgins, pretrained GloVe embeddings, ...) were tested. Classification was performed using different classifiers (Logistic Regression, Naive Bayes, Support Vector Machine, Convolutional Neural Networks, Long Short Term Memory Networds) and performance was evaluated.
This project is part of the EPFL CS-433: Machine learning class.

## Folders
* data: folder containing 10% of the raw data, since the size of the whole data is too large, and also the cleaned one using the Ruby preprocessing
* dico: folder containing a dictionnary (typo-corpus-r1.txt) of common english spelling errors, found on internet: http://luululu.com/tweet/
* fig: folder containing the figures made with our data
* preprocessing: folder containing python files to preprocess the data. There is several files since there is several way to preprocess the raw data
* results: folder containing our submissions created once code is run
* src: folder containing all pieces of codes required to recreate the analysis described in our report

## Run
* Run preprocessing of data: 
  - run processing_tweet.py
* Run SVM classification : 
  - run split_data.py 
  - run embed_main.py 
  - run train_main.py
* Run CNN/LSTM classification :
  - run main_DNN.py
* Run AI Crowd submission:
  - run submissions.ipynb

## Necessary libraries
* numpy
* matplotlib
* pandas
* os
* re
* keras
* sklearn
* pickle
* csv
* itertools
* string
* glove_python 
* gensim
* tensorflow
* tempfile
* random
* scipy

## Aditionnal Ressources

GloVe pretrained tweet embeddings should be downloaded from the https://nlp.stanford.edu/projects/glove/ website and stored in the data folder.

## Collaborators : BIZEUL Alice, DERVAUX Juliane, SIERRO Florent
* Bizeul Alice:	 	alice.bizeul@epfl.ch
* Dervaux Juliane:	juliane.dervaux@epfl.ch
* Sierro Florent:	florent.sierro@epfl.ch
