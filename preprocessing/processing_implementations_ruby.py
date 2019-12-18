# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:56:06 2019

@author: Florent
"""
# -*- coding: utf-8 -*-

import re
import itertools
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 


#%% This section is inspired from this https://gist.github.com/ppope/0ff9fa359fb850ecf74d061f3072633a with some modifications 

"""
Script for preprocessing tweets by Jeffrey Pennington (https://gist.github.com/ppope/0ff9fa359fb850ecf74d061f3072633a)
with small modifications by Florent Sierro

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    """
    Suppres the hastags before the words and add a hashtag tag , thus the words are now recognized as correctly spelled words
    e.g. "#love" -> "love <hashtag>"  
            
    INPUT:
        |--- text: [str] original token
    OUTPUT:
        |--- result: [str] the token without the symbol "#" and the special tag added
    """
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result


def tokenize(text,remove_bracket=False):
    """
    Replace the tweet writing to normal english sentence
            
    INPUT:
        |--- tweet: [str] original tweet yet preprocess
        |--- remove_bracket: [bool] boolean to activate or not the removal of words <url> and <user> present in the tweets
    OUTPUT:
        |--- text: [str] tweet with all Twitter formulations replaced by normal English word and special tags that emphasize feelings
    """
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    
    # Added emojis to handle the bigger writing styles of emojis
    text = re_sub(r":-\||:\||=\||>:\|","<neutralface>") 
    text = re_sub(r"\)\':|:\'\(|;\'\(|:\'{","<sadface>")
    text = re_sub(r"\/8|:\(|:\/|:\\|\/:|\/\':|\/\';|\/-:|\[:<|:-\/|:\'\/|:\[|:{|:-\[|:-\\|\\8|\\:|\\=|\/=|]:|];|]=|\)=|=\(|;\/|=\[|=\\|={|=]|=\/|;\\|>=\(|>:\/|:o\(|\/o8|\/o:","<sadface>")               
    text = re_sub(r"\(8|\(\':|\(=|:]|:\)|\(:|:\'\)|:}|:-}|:-]|\[8|\[:|\[=|=\)|=}|>=\)|:o\)|\(o:|\(-8|:\']|:-\)","<smileface>")
    text = re_sub(r"\[;|\[\';|;\'\)|;]|;o\)|;}|\(\';","<smileface>")
    text = re_sub(r":-p|:\'p|;-p|:p|;p|=p","<lolface>")     
    text = re_sub(r":@|:-@|;@|=@","<angryface>")
    text = re_sub(r":-\*|:\*|:×","<kissface>")                  
    
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smileface>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"\.\.\.","<suspensionpoints>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    
    if remove_bracket: 
        text = re_sub('<user>', '') 
        text = re_sub('<url>', '')

    return text.lower()

#%%

ps = PorterStemmer()
wnl = WordNetLemmatizer()


def appostophes_replacing(tweet,IS=True):
    """
    Replace the contractions appostophes by the whole word to avoid any duplicate words.
    The contractions have been expand, for instance "you're" became "you are". 
    The "'s" contraction is specially treated because it can expand to "is" or "has", and even remain for possessive, it possible to disactivate this special expansion
            
    INPUT:
        |--- tweet: [str] the expression to expand
        |--- IS: [boolean] to activate or not the "'s" expansion
    OUTPUT:
        |--- reformed: [str] a string composed of the 2 words separated
    """
    if IS:
        tweet = re.sub(r"\'s", " \'s", tweet) #since "'s" can stand for "is" , "has" or possessive "'s"
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"won\'t", "will not", tweet)
    tweet = re.sub(r"\'m", " \'m", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet) 
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    
    APPOSTOPHES = {"'s":"is", 
                   "'m":"am", 
                   "'re":"are", 
                   "n't":"not",
                   "'ll":"will",
                   "'ve":"have",
                   "'d":"would"}
    words = tweet.split()
    reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
    reformed = " ".join(reformed)
    
    return reformed


def load_dico():
    """
    Load a dictionnary of common english spelling errors
            
    OUTPUT:
        |--- dictionary: [dict] dictionary with typo as key and corrected words as value
    """
    dictionary = {} 
    with open("../dico/" + "typo-corpus-r1.txt", mode='rt') as file:
        for line in file:
          typo, original, _, _, _, _ = line.rstrip('\n').split('\t')
          dictionary.update({typo:original})
    return dictionary


def spell_check(word,dictionary):
    """
    Commonly misspeled words are corrected to better handle non-sens/non-existing words
    The spelling mistakes are changed according to a typo-corpus
            
    INPUT:
        |--- word: [str] misspeled and correct english words 
        |--- dictionary: [dict] dictionary with typo as key and corrected words as value
    OUTPUT:
        |--- word: [str] corrected english words
    """
    if word in dictionary:
         word = dictionary[word] 
    return word


def remove_repetitions_letter(word):
    """ 
    Reduce the repetitions character, either in the middle or at the end of a word as well as punctuation mark, to one character. 
    e.g. "niceeeeee" -> "nice" or "cooooool" -> "cool" 
    
    INPUT:
        |--- word: [str] word tokens or punctuation tokens
    OUTPUT:
        |--- word: [str] token reduced to single caractere repetition 
    """ 
    word = ''.join(''.join(s)[:1] for _, s in itertools.groupby(word))    
    return word
    

def remove_tweet_repetition(wf, previous_tweet, tweet):
    """
    If the same tweet follows, the repetition are suppressed in order to not emphasize the token in the tweet
        
    INPUT:
        |--- tweet: [str] the tweet that have been cleaned (i.e. preprocessed)
        |--- previous_tweet: [str] the previous twwet cleane, set to '' for the first tweet
        |--- wf: the writing file 
    OUTPUT:
        |--- tweet [str]
        |--- previous_tweet [str]
        |--- wf: the writing file with the new tweet added if not the same as before
    """
    if tweet == previous_tweet: # or tweet=='':
        tweet=tweet
    else: 
        previous_tweet = tweet                              
        wf.write(''.join(tweet) +'\n')    
    return wf, previous_tweet, tweet


def cleaning_tweet_ruby(file_name, file_name_out, remove_bracket, stemming=False, lemmatization=False):
    """
    Includes all the previous function in the right order to clean the tweets and write a new file cleaned
            
    INPUT:
        |--- file_name: [str] a string including the file path and name to preprocess 
        |--- file_name_out: [str] string to know with preprocessing options was chosen 
        |--- remove_bracket: [bool] boolean to activate or not the removal of words <url> and <user> present in the tweets
        |--- stemming: [bool] boolean to activate or not the Stemming
        |--- lemmatization: [bool]  boolean to activate or not the Lemmatization
    """
    
    dictionary = load_dico()
    previous_tweet=''
        
    with open(file_name, mode='rt', encoding='utf-8') as rf: 
        with open(file_name[:-4] + file_name_out + '.txt', mode='wt', encoding='utf-8') as wf:    
        
            for tweet in rf:
                if "test" in file_name:
                    line =  tweet.strip()
                    line = appostophes_replacing(line,True)
                    ID = tweet.strip().split(',')[0]+','
                    line = line.replace(ID, "")
                else:
                    line =  tweet.strip()
                    line = appostophes_replacing(line,True)

                line = tokenize(line,remove_bracket)
                
                words = line.split()
                for w in range(len(words)):
                    words[w] = remove_repetitions_letter(words[w])
                    words[w] = spell_check(words[w],dictionary)
                    if stemming:
                        words[w] = ps.stem(words[w]) 
                    if lemmatization:
                        words[w] = wnl.lemmatize(words[w])                                   
                line=' '.join(words).strip()
                line=' '.join(line.split()) 


                if "test" in file_name:
                    wf.write(''.join(line) +'\n')
                else:
                    wf, previous_tweet, line = remove_tweet_repetition(wf, previous_tweet, line)           
           
            
def processing_tweet_ruby(full, file_name_out, remove_bracket, stemming, lemmatization):
    """
    Call the function cleaning_tweet with the right files wanting to be preprocessed 
            
    INPUT:
        |--- full: [bool] boolean to clean the full train files 
        |--- filn_name_out: [str] string to know with preprocessing options was chosen 
        |--- remove_bracket: [bool] boolean to activate or not the removal of words <url> and <user> present in the tweets
        |--- stemming: [bool] boolean to activate or not the Stemming
        |--- lemmatization: [bool]  boolean to activate or not the Lemmatization
    """

    path_data = '../data/twitter-datasets/'
    
    if full:
        file_name_pos = path_data + 'train_pos_full.txt'
        file_name_neg = path_data + 'train_neg_full.txt'
    else:
        file_name_pos = path_data + 'train_pos.txt'
        file_name_neg = path_data + 'train_neg.txt'  

    cleaning_tweet_ruby(file_name_pos, file_name_out, remove_bracket, stemming, lemmatization)
    print("Half done")
    cleaning_tweet_ruby(file_name_neg, file_name_out, remove_bracket, stemming, lemmatization)
          
