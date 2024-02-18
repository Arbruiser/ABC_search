#!/usr/bin/python3

from flask import Flask, render_template, request
from bs4 import BeautifulSoup 
import matplotlib.pyplot as plt
import matplotlib as mlp
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
import os



mlp.use('Agg')
fpath='medical_document.txt'
with open(fpath, 'r', encoding='utf-8') as f:
    documents = f.read()

documents = documents.split('\n\n')  # probably should be done within search.py as we wouldn't have to do the same thing twice and make it slower
articles = [doc.split('\n', 1)[1].strip() for doc in documents] # splits the document at the first occurrence of a newline character. This separates the link from the rest of the document.
# `[1]` gets the second part of the split (i.e., everything after the link). `.strip()` removes any whitespace from the text.

# make it a dictionary where key is the name of the article and value are the words
articles_dict = {} 
for doc in articles:
    title, text = doc.split('\n', 1)  # Splits the document at the first occurrence of a newline character
    text = text.strip()  # Removes whitespace from the text
    articles_dict[title] = text  # Add title as key and text as value to the dictionary

print('First Key:', list(articles_dict.keys())[0], '\nFirst Value:', list(articles_dict.values())[0]) # for debugging

frequencies = {} # list of dictionaries where keys are words and values counts of them 
for key, value in articles_dict.items():
    words = nltk.word_tokenize(value.lower())
    frequencies[key] = dict(nltk.FreqDist(words))

# Combine all articles and compute frequency distribution
all_words = []
for text in articles_dict.values():
    for word in nltk.word_tokenize(text.lower()):  
        if word.isalpha() and word not in stop_words:  # remove non-alphabetic tokens and stop words
            all_words.append(word)

all_words_freq = nltk.FreqDist(all_words)

# Get the n most common words and their frequencies
most_common_all = dict(all_words_freq.most_common(30))

# Create bar plot
plt.figure(figsize=(10,5))
plt.bar(most_common_all.keys(), most_common_all.values())
plt.title('Word Frequency for All Documents')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.25)
plt.savefig('word_freq_all.png')



