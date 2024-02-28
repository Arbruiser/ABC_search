#!/usr/bin/python3
# this script plots the frequency over our recent medical documents excluding stop words
import matplotlib.pyplot as plt
import matplotlib as mlp
import nltk

# nltk.download('stopwords') # uncomment these if you don't yet have them
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
# nltk.download('punkt')

mlp.use("Agg")
fpath = "Site/NER_and_plotting/NERed_recent_medical_documents.json"
with open(fpath, "r", encoding="utf-8") as f:
    documents = f.read()

documents = documents.split("\n\n")
articles = [
    doc.split("\n", 1)[1].strip() for doc in documents
]  # splits the document at the first occurrence of a newline character. This separates the link from the rest of the document.
# `[1]` gets the second part of the split (i.e., everything after the link). `.strip()` removes any whitespace from the text.

# make it a dictionary where key is the name of the article and value are the words
articles_dict = {}
for doc in articles:
    title, text = doc.split(
        "\n", 1
    )  # Splits the document at the first occurrence of a newline character
    text = text.strip()  # Removes whitespace from the text
    articles_dict[title] = text  # Add title as key and text as value to the dictionary

frequencies = {}  # list of dictionaries where keys are words and values counts of them
for key, value in articles_dict.items():
    words = nltk.word_tokenize(value.lower())
    frequencies[key] = dict(nltk.FreqDist(words))

# Combine all articles and compute frequency distribution
all_words = []
for text in articles_dict.values():
    for word in nltk.word_tokenize(text.lower()):
        if (
            word.isalpha() and word not in stop_words
        ):  # remove non-alphabetic tokens and stop words
            all_words.append(word)

all_words_freq = nltk.FreqDist(all_words)

# Get the n most common words and their frequencies
most_common_all = dict(all_words_freq.most_common(30))

# Create horizontal bar plot
plt.figure(figsize=(10, 5))
plt.bar(most_common_all.keys(), most_common_all.values(), color="darkorchid")
plt.title("Word Frequency for All Documents")
plt.grid(axis="y", linestyle="-", linewidth=0.2)  # Adds thin horizontal gridlines
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.25)
plt.savefig(f"Site/NER_and_plotting/Plots/word_freq_no_NER_horizontal.png")

# vertical bar plot
plt.figure(figsize=(10, 8))
plt.barh(
    list(most_common_all.keys()), list(most_common_all.values()), color="darkorchid"
)  # can change it to vertical plot
plt.title("Word Frequency for All Documents")
plt.grid(axis="x", linestyle="-", linewidth=0.2)  # Adds thin horizontal gridlines
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.gca().invert_yaxis()  # Inverts the order
plt.subplots_adjust(bottom=0.1)
plt.savefig(f"Site/NER_and_plotting/Plots/word_freq_no_NER_vertical.png")
