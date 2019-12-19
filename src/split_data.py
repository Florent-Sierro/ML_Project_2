
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split 
import os

# Load data 
root = '../data/'
train_pos = np.loadtxt(root+'train_pos_ruby.txt',delimiter='\n',dtype = str)
train_neg = np.loadtxt(root+'train_neg_ruby.txt',delimiter='\n',dtype = str)
train = np.concatenate((train_pos,train_neg))
labels = np.concatenate((np.ones((train_pos.shape[0],)), -1* np.ones((train_neg.shape[0],))))

# Split train-val
train_tweets, val_tweets, train_labels, val_labels = train_test_split(train, labels, test_size=0.1, random_state=42, shuffle=True, stratify=labels)
 
# Save train
folder = root +'/train'
if not os.path.exists(folder): os.mkdir(folder)
df_train = pd.DataFrame(data= train_tweets, columns = ['Tweets']).to_csv(root +'/train.csv',index = False)
with open(root+'/train_labels.pkl', 'wb') as f:
    pickler = pickle.Pickler(f)
    pickler.dump(train_labels)
    f.close()

# Save validation
folder = root +'/val'
if not os.path.exists(folder): os.mkdir(folder)
df_val = pd.DataFrame(data= val_tweets, columns = ['Tweets']).to_csv(root +'/val.csv',index = False)
with open(root+'/val_labels.pkl', 'wb') as f:
    pickler = pickle.Pickler(f)
    pickler.dump(val_labels)
    f.close()

print("Number of tweets : {:0.4f}".format(len(train_tweets)))
print("Proportion class 1 in train : {:0.4f}".format(np.sum(train_labels == 1)/train_labels.shape[0]))
print("Proportion class 1 in val : {:0.4f}".format(np.sum(val_labels== 1)/val_labels.shape[0]))