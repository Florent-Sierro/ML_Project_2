#!/bin/bash
from processing_implementations_ruby import cleaning_tweet_ruby
from processing_implementations_extensive import cleaning_tweet_extensive

file_path = "../data/twitter-datasets/"
file_to_clean = "train_flo.txt"


full= False
lemmatization=False 
stemming=False

#%% Ruby
remove_bracket=False 
Name_out = "_Ruby"
cleaning_tweet_ruby(file_path + file_to_clean, Name_out, remove_bracket, stemming, lemmatization)


#%% Extensive
only_words = True
stopword = True
Name_out = "_Extensive"
cleaning_tweet_extensive(file_path + file_to_clean, Name_out, only_words, stopword, stemming, lemmatization)


