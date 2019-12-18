# -*- coding: utf-8 -*-

import re
import itertools
import string
from html.parser import HTMLParser
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
#nltk.download('stopwords') #to comment once already download
#nltk.download('wordnet') #to comment once already download


html_parser = HTMLParser()
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


def tweet_to_phrase(word):
    """
    Suppres the hastags before the words, thus the words are now recognized as correctly spelled words
    e.g. "#love" -> "love <hashtag>"  
            
    INPUT:
        |--- word: [str] original token
    OUTPUT:
        |--- word: [str] the token without the symbol "#" 
    """
    word = re.sub('^#', '', word)
    word = re.sub('#', '', word) 
   
    return word
  
def emoji_substitution (word):
    """
    Transform emjis punctuation into tag-words such as angry, happy, kiss faces, and hearts
            
    INPUT:
        |--- word: [str] original token
    OUTPUT:
        |--- word: [str] tweet with emojis formulations replaced by tag-words
    """    
    word = re.sub (r"<3","heartsmiley", word)
    word = re.sub (r"\(8|\(\':|\(=|:]|:\)|\(:|:\'\)|:}|:-}|:-]|\[8|\[:|\[=|=\)|=}|>=\)|:o\)|\(o:|\(-8|:\']|:-\)","happysmiley",word)
    word = re.sub (r"\[;|\[\';|;\'\)|;]|;o\)|;}|\(\';","winksmiley",word)
    word = re.sub (r":-\*|:\*|:×","kisssmiley",word)
    word = re.sub (r":-p|:\'p|;-p|:p|;p|=p","tonguesmiley",word)               
    word = re.sub (r":d|:-d|;d|;-d|=d|>:d","laughsmiley",word)
    word = re.sub (r":@|:-@|;@|=@","angrysmiley",word)              
    word = re.sub (r":-\||:\||=\||>:\|","neutralsmiley",word) 
    word = re.sub (r"\)\':|:\'\(|;\'\(|:\'{","cryingsmiley",word)
    word = re.sub (r"\/8|:\(|:\/|:\\|\/:|\/\':|\/\';|\/-:|\[:<|:-\/|:\'\/|:\[|:{|:-\[|:-\\|\\8|\\:|\\=|\/=|]:|];|]=|\)=|=\(|;\/|=\[|=\\|={|=]|=\/|;\\|>=\(|>:\/|:o\(|\/o8|\/o:","sadsmiley",word)               

    return word
      
       
def remove_bracket(word):
    """
    Remove term between angle bracket which are <user> and <url> and all other expression in brackets
    
    INPUT:
        |--- word: [str] original token
    OUTPUT:
        |--- word: [str] corrected token, either unchanged or empty
    
    """ 
    word = re.sub('<user>', '', word) 
    word = re.sub('<url>', '', word)
    word = re.sub(r'<([^>]+)>', '',word)    #Remove all words between <...>
    word = re.sub(r'_([^>]+)_', '',word)    #Remove all words between _..._
        
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
    

def remove_numbers(word):
    """
    Remove words composed only of number, 
    Remove words beginning with number, such as hours 20:30, 20h30, 10pm
    Keep words having number in it, such as x15, me2, <3

    INPUT:
        |--- word: [str] original token
    OUTPUT:
        |--- word: [str] corrected token, either unchanged or empty
    """
    if word.isnumeric():
        word=''
    if re.match(r'^[0-9]',word):
        word=''
       
    return word           
       
 
def remove_ponctuation(word):
    """
    Clear all single ponctuation but keep the smiley which need more than one ponctuation character ! 
    
    INPUT:
        |--- word: [str] original token
    OUTPUT:
        |--- word: [str] corrected token, either unchanged or empty
    """
    if len(word)==1: 
        word = word.translate(str.maketrans('', '', string.punctuation))      
   
    return word   


def remove_single_char(word):
    """
    Clear all single letter character that can remain ! 
    
    INPUT:
        |--- word: [str] original token
    OUTPUT:
        |--- word: [str] corrected token, either unchanged or empty
    """
    if len(word)==1: 
        word=''

    return word


def remove_stop_words(word):
    """
    Remove stopwords such as for, by, a, an , and, ...
    
    INPUT:
        |--- word: [str] original token
    OUTPUT:
        |--- word: [str] corrected token, either unchanged or empty
    """
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    if word in stop_words:
        word=''

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


def cleaning_tweet_extensive(file_name, file_name_out, only_words=True, stopword=False, stemming=True, lemmatization=True):
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
                    words = line.split()
                else:
                    line =  tweet.strip()
                    line = appostophes_replacing(line,True)
                    words = line.split()
                    ID=""
                
                for w in range(len(words)):
                    
                    words[w] = tweet_to_phrase(words[w])
                    words[w] = remove_bracket(words[w])
                    words[w] = emoji_substitution(words[w])
                    words[w] = remove_repetitions_letter(words[w])
                    words[w] = spell_check(words[w],dictionary)
                    
                    if only_words:
                        words[w] = remove_numbers(words[w])
                        words[w] = remove_ponctuation(words[w])
                        
                    if stopword:    
                        words[w] = remove_stop_words(words[w])
                    
                    if stemming:
                        words[w] = ps.stem(words[w]) 
                    
                    if lemmatization:
                        words[w] = wnl.lemmatize(words[w])
                        
                    words[w]= remove_single_char(words[w])
                                    
                tweet=' '.join(words).strip()
                tweet=' '.join(tweet.split()) 

                if "test" in file_name:
                    wf.write(''.join(tweet) +'\n')
#                    wf.write(ID+''.join(tweet) +'\n')
                else:
                    wf, previous_tweet, tweet = remove_tweet_repetition(wf, previous_tweet, tweet)           
           
                
def processing_tweet_extensive(full, file_name_out, only_words, stopword, stemming, lemmatization):
    """
    Call the function cleaning_tweet with the right files wanting to be preprocessed 
            
    INPUT:
        |--- full: [bool] boolean to clean the full train files 
        |--- file_name_out: [str] string to know with preprocessing options was chosen 
        |--- only_words: [bool] boolean to suppress all the numerical expression and punctuation
        |--- stopword: [bool] boolean      
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

    cleaning_tweet_extensive(file_name_pos, file_name_out, only_words, stopword, stemming, lemmatization)
    print("Half done")
    cleaning_tweet_extensive(file_name_neg, file_name_out, only_words, stopword, stemming, lemmatization)

          
                
                
                
                
        
      