#!/usr/bin/python3
# this script plots the frequency over our recent medical documents excluding stop words
import matplotlib.pyplot as plt
import matplotlib as mlp
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

mlp.use("Agg")
fpath = "medical_document.txt"
with open(fpath, "r", encoding="utf-8") as f:
    text = f.read()

# Use the nltk word_tokenize function to tokenize the text into words
all_words = []
for word in nltk.word_tokenize(text): # tokenizes text and then checks if tokens are words and not stop words
    word = word.lower()
    if (word.isalpha() and word not in stop_words): 
        all_words.append(word)

all_words_freq = nltk.FreqDist(all_words)

# Get the n most common words and their frequencies
most_common_all = dict(all_words_freq.most_common(30))

# Create horizontal bar plot
plt.figure(figsize=(10, 5))
plt.bar(most_common_all.keys(), most_common_all.values(), color="darkorchid")
plt.title("Simple word frequency, no NER")
plt.grid(axis="y", linestyle="-", linewidth=0.2)  # Adds thin horizontal gridlines
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.25)
plt.savefig(f"Site/static/Plots/word_freq_no_NER_horizontal.png")

# vertical bar plot
plt.figure(figsize=(10, 8))
plt.barh(list(most_common_all.keys()), list(most_common_all.values()), color="darkorchid")  # can change it to vertical plot
plt.title("Simple word frequency, no NER")
plt.grid(axis="x", linestyle="-", linewidth=0.2)  # Adds thin horizontal gridlines
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.gca().invert_yaxis()  # Inverts the order
plt.subplots_adjust(bottom=0.1)
plt.savefig(f"Site/static/Plots/word_freq_no_NER_vertical.png")
