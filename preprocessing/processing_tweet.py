#!/bin/bash
from processing_implementations_ruby import processing_tweet_ruby
from processing_implementations_extensive import processing_tweet_extensive

file_path = "../data/twitter-datasets/"
full = False

#%% Ruby
remove_bracket = False 
stemming = False
lemmatization = False 

Name_out = "_ruby"
processing_tweet_ruby(full, Name_out, remove_bracket, stemming, lemmatization)


#%% Extensive
only_words = True
stopword = True
stemming = True
lemmatization = False 

Name_out = "_extensive"
processing_tweet_extensive(full, Name_out, only_words, stopword, stemming, lemmatization)


