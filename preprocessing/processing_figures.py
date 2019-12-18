# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:41:51 2019

@author: Florent
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




with open("../data/train_pos_full.txt", mode='rt', encoding='utf-8') as rf: 

    indice=[]
    size_pos=[]
    char_pos=[]
    for tweet in rf:
        line =  tweet.strip()
        char_pos.append(len(line))
        words = line.split()
        size_pos.append(len(words))



with open("../data/train_neg_full.txt", mode='rt', encoding='utf-8') as rf: 
    
    size_neg=[]
    char_neg=[]
    for tweet in rf:
        line =  tweet.strip()
        char_neg.append(len(line))
        words = line.split()
        size_neg.append(len(words))
        

        

#%% Plot of number of char
        
        
char_tweet = [char_pos, char_neg]

fig, axes = plt.subplots(figsize=(6, 6))
axes.boxplot(char_tweet)
title_0 = "#Characters in tweet"
axes.set_title(title_0)
axes.set_ylabel('Number of character')
labels = ["Positive","Negative"]
axes.set_xticklabels(labels)
plt.show()
fig.savefig('../fig/Tweet_boxplot_char')


char_tweet_T = [char_neg, char_pos]
df = pd.DataFrame(char_tweet_T, index=["Negative","Positive"])
plt.subplots(figsize=(14, 6))
boxplot = df.T.boxplot(vert=False, fontsize=16, grid=False, whiskerprops = dict(linestyle='-',linewidth=2.0, color='blue'))
boxplot.set_xlabel('Number of character',fontsize=16)
plt.show()
plt.savefig('../fig/Tweet_boxplot_char_horizontal')


#%% Plot of number of words
        
        
size_tweet = [size_pos, size_neg]

fig, axes = plt.subplots(figsize=(6, 6))
axes.boxplot(size_tweet)
title_0 = "#Token in tweet"
axes.set_title(title_0)
axes.set_ylabel('Number of token')
labels = ["Positive","Negative"]
axes.set_xticklabels(labels)
plt.show()
fig.savefig('../fig/Tweet_boxplot')


#%%

           
number_bin = 20
counts_pos, bins = np.histogram(size_pos, bins=20, range=(1, 101))
counts_neg, bins = np.histogram(size_neg, bins=20, range=(1, 101))     
labels = ['1-10', '11-20', '21-30', '31-40','41-50','51-60','61-70', '71-80', '81-90', '91-100']
lab = ['1-5', '6-10', '11-15', '16-20','21-25','26-30','31-35', '36-40', '40+']

pos=[]
neg=[]
sum_pos = 0
sum_neg = 0
for i in range(len(counts_pos)):
    if i<8:
        pos.append(counts_pos[i])
        neg.append(counts_neg[i])
    else:
        sum_pos = sum_pos + counts_pos[i]
        sum_neg = sum_neg + counts_neg[i]
pos.append(sum_pos)
neg.append(sum_neg)

x = np.arange(len(lab))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width/2, pos, width, label='Positive')
rects2 = ax.bar(x + width/2, neg, width, label='Negative')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of tweets', fontsize=14)
ax.set_xlabel('Number of token in a tweet', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(lab, fontsize=16)
ax.legend(fontsize=16)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

fig.tight_layout()

plt.show()
fig.savefig('../fig/Tweet_barplot')

