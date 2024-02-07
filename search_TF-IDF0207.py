#!/usr/bin/env python3
"""
Update: 2024-02-07
1. Integrate the query function, now the query will ask user to choose between Boolean and TF-IDF search.
2. Added colorama to highlight the context of the query (only on boolean search result for testing).

Future work:
- If the word in the query is between quotes, do not stem the word.

Update: 2024-02-03
(Arthur) Added Porter stemmer. Now we use stemmed documents for making our matrices. User query is also stemmed. 
---Now there is a scary for loop that basically splits the documents into stemmed and unstemmed documents. Stemmed is for our TF-IDF search.
Unstemmed is for showing the context. Now documents is a list of strings, but the for loop in addition to this makes lists of lists where
each emelement is its own sentence. This is necessary for showing the context because we have to search with our stemmed query in the stemmed 
matrix, and then take the number of document and sentence and show it in the UNstemmed version. It works. 
---I also added a toy document about sneezing so that you can play with queries "sneezes", "sneezed", "sneezing" which are all stemmed into 'sneez'
---I also fixed the issue of our context looking ugly, it was because of new lines that stayed there after the links got deleted.
---Now we also show the score of the matching doc. We can translate it into plain English with a bunch of if statements
like "if score>0.3 then "score very high!" and so on.
--- If the query contains multiple words, the context function will look for ANY of the query words in the text in order 
and output only the first 3 matches. Try running "sneezing is not an illness". 

Work before the class:
--- Let user choose if to use Boolean or TF-IDF search. Maybe with the first input asking for B (Boolean) or T (TF-IDF)?  
--- Can implement indexing of the sentences in the matched document to output the highest scoring sentences first. 
--- Can implement highlighting the matched word in the sentence in the same fashion as looking for unstemmed docs. We turn each sentence into
stemmed and unstemmed lists of words. Search for the stemmed query word in the stemmed lists and put *...* around the word from the same number of document
and the same number of sentence and the same number of word 

Update: 2024-02-01
1. Now the search result only returns the first 3 matches.
2. Added more documents to the corpus. Now is 13.

Update: 2024-01-29

1. Now the search result includes the document's title, link, and the context of the query.
The result is in below format:
<Document Title>
<Document Link>
... <The sentence includes the query keyword> ...
Note: The context sentence only include the first keyword in the query.

2. Now if the query is invalid, the program will not crash. Instead, it will print "Invalid query, please try again." and ask for another query.

Minor changes:

1. Cleaned up the code a little bit.
2. Removed the unnecessary variables and functions.

Future work:

- Will add more documents to the corpus.
- Try to rank the documents (maybe by word frequency or other methods). Done
- Make logic that will allow searching for multiple words with some unknown words (Arthur) 

Challenges:
- If the query has more than one keyword, try to show the sentence that includes all the keywords. Done
- User doesn't have to type "and" or "AND" to use the AND operand. They can just type a space between two keywords. Done
- Using embedding models to do fuzzy search. 

"""
# import dependencies
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from colorama import Fore, Style

stemmer = PorterStemmer()  # let's use the basic stemmer

# import the documents
with open("medical_document.txt", "r", encoding="utf-8") as f:
    content = f.read()

documents = content.split("\n\n")  # makes a list of our string documents

# Get the first line and second line of each doc
httplinks = []
titles = []
for i in range(len(documents)):
    httplinks.append(documents[i].split("\n")[0])
    titles.append(documents[i].split("\n")[1])

# Remove the httplinks and titles from the documents. Also delete the new lines.
for i in range(len(documents)):
    documents[i] = (
        documents[i].replace(httplinks[i], "").replace("\n", " ").strip()
    )  # replace remaining newline characters with space.
    documents[i] = (
        documents[i].replace(titles[i], "").replace("\n", " ").strip()
    )  # replace remaining newline characters with space.


# Loop where we split the documents into sentences and stem. Brace yourselves.
stemmed_documents_lists = []  # list of lists
stemmed_documents = []  # list of strings
documents_lists = []  # again list of lists

for document in documents:
    temp_sentences_unstemmed = re.split(
        "([.!?])\s+", document
    )  # here we split the sentences with delimiters being their own elements in the list.
    # However, that's unnecessary now because in our doc each line is a sentence. We don't know if it stays that way, so let's keep it.
    # next line joins the delimiters with the previous sentence (don't worry about it)
    sub_doc = [
        temp_sentences_unstemmed[i]
        + (
            temp_sentences_unstemmed[i + 1]
            if i + 1 < len(temp_sentences_unstemmed)
            else ""
        )
        for i in range(0, len(temp_sentences_unstemmed), 2)
    ]
    documents_lists.append(sub_doc)

    document = document.split()  # split the doc into words to prepare it for stemming
    stemmed_document = " ".join(
        [stemmer.stem(word) for word in document]
    )  # stem and join the text back
    stemmed_documents.append(
        stemmed_document
    )  # this produces a list of strings that we use for TF-IDF
    temp_sentences_stemmed = re.split(
        "([.!?])\s+", stemmed_document
    )  # the same splitting with .!? as delimiters
    sub_stemmed_doc = [
        temp_sentences_stemmed[i]
        + (temp_sentences_stemmed[i + 1] if i + 1 < len(temp_sentences_stemmed) else "")
        for i in range(0, len(temp_sentences_stemmed), 2)
    ]
    stemmed_documents_lists.append(
        sub_stemmed_doc
    )  # this now produces a list of our docs, where each doc is a list of sentences. This is only for context.


# Make a boolean matrix of our terms and convert it to dense
cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(documents)
dense_matrix = sparse_matrix.todense()

td_matrix = dense_matrix.T  # .T transposes the matrix

t2i = cv.vocabulary_  # The dictionary for query

# Boolean operands
d = {
    "and": "&",
    "AND": "&",
    "or": "|",
    "OR": "|",
    "not": "1 -",
    "NOT": "1 -",
    "(": "(",
    ")": ")",
}


def rewrite_token(t):
    # when we detect and/or/not we replace them with our operands
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))


def rewrite_query(query):  # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())


sparse_td_matrix = (
    sparse_matrix.T.tocsr()
)  # makes the matrix ordered by terms, not documents

# TF-IDF
tfidf = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
tf_idf_matrix = tfidf.fit_transform(stemmed_documents).T.tocsr()  # using stemmed docs!


def search_start(query_string):
    # Vectorize query string
    query_vec = tfidf.transform([query_string]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, tf_idf_matrix)

    # Rank hits
    ranked_scores_and_doc_ids = sorted(
        zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True
    )
    ranked_scores_and_doc_ids = ranked_scores_and_doc_ids[:3]

    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
        print()
        print(
            titles[doc_idx], "--- Its score is:", round(score, 2)
        )  # print the title and score
        print(httplinks[doc_idx])  # print the link

        # for each sentence in the stem-docs, check if ANY word in the query is in the sentence, print corresponding original sentence
        count = 0
        for j, stemmed_sentence in enumerate(stemmed_documents_lists[doc_idx]):
            if any(word in stemmed_sentence for word in query_string.split()):
                print("..." + documents_lists[doc_idx][j] + "...")
                count += 1
                if count == 3:  # limits the number of contexts to 3
                    break


# QUERY
def query():
    while True:
        bort = input(
            "Do you want to use Boolean search (b) or TF-IDF search?(t): "
        ).lower()
        if bort == "b":  # using boolean search
            user_query = input("Hit enter to exit. Your query to search: ")
            if user_query == "":
                break
            print("Query: '" + user_query + "'")
            hits_matrix = eval(rewrite_query(user_query))  # the query
            hits_list = list(hits_matrix.nonzero()[1])

            if len(hits_list) > 3:
                hits_list = hits_list[:3]

            for doc_idx in hits_list:
                print()
                print(titles[doc_idx])  # print the title
                print(httplinks[doc_idx])  # print the link

                if len(user_query.split()) == 1:
                    # regex to find the sentence with the query
                    pattern = (
                        r"([^.!?]*" + user_query + r"[^.!?]*[.!?])"
                    )  # matches the sentence with .!? as delimiters
                    match = re.search(pattern, documents[doc_idx], re.IGNORECASE)
                    print(
                        Fore.BLUE
                        + "... "
                        + match.group(0).strip()
                        + " ..."
                        + Style.RESET_ALL
                    )
                    print()
                else:  # only show the context of the first term in the query
                    pattern = r"([^.!?]*" + user_query.split()[0] + r"[^.!?]*[.!?])"
                    match = re.search(pattern, documents[doc_idx], re.IGNORECASE)
                    print(
                        Fore.BLUE
                        + "... "
                        + match.group(0).strip()
                        + " ..."
                        + Style.RESET_ALL
                    )
                    print()

        elif bort == "t":
            user_query = input("Hit enter to exit. Your query to search: ")
            if user_query == "":
                break
            stemmed_query = " ".join(stemmer.stem(word) for word in user_query.split())
            print("stemmed query:", stemmed_query)
            search_start(stemmed_query)
        elif bort == "":
            break
        else:
            print("Invalid input. Please try again.")
            print()  # if the user hits enter, the program will exit


if __name__ == "__main__":
    query()

# print("Hit enter to exit.")
# while True:
#     user_query = input("\nYour query to search: ")
#     stemmed_query = " ".join(stemmer.stem(word) for word in user_query.split())

#     if user_query == "":
#         break

#     print("stemmed query:", stemmed_query)  # for debugging

#     try:
#         search_start(stemmed_query)
#     except:
#         print("Invalid query, please try again.")
