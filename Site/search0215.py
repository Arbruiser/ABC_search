#!/usr/bin/env python3
"""
Update: 2024-02-08
1. Added the fuzzy search using sentence-transformers. Now the user can choose between Boolean, TF-IDF, and fuzzy search.
2. When user's query contains quotes, the program will not stem the word in the quotes. 

Bug fixed:
The top 3 results now will be unique.

Future work:
Improve the exact match logic. Now it only checks if there are any quote symbols in the query.
We need the program to match the word between quotes exactly rather than just checking if there are quotes in the query.

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

Please refer to Readme.md for ancient update logs.
"""
# import dependencies
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from colorama import Fore, Style

stemmer = PorterStemmer()  # let's use the basic stemmer

# import the documents
with open("./medical_document.txt", "r", encoding="utf-8") as f:
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


# Segment the documents into sentences
# Loop where we split the documents into sentences and stem. Brace yourselves.
stemmed_documents_lists = []  # list of lists
stemmed_documents = []  # list of strings
documents_lists = []  # again list of lists


for document in documents:
    temp_sentences_unstemmed = re.split("([.!?])\s+", document)  # here we split the sentences with delimiters being their own elements in the list. However, that's unnecessary now because in our doc each line is a sentence. We don't know if it stays that way, so let's keep it.
    # next line joins the delimiters with the previous sentence (don't worry about it)
    sub_doc = [temp_sentences_unstemmed[i]+ (temp_sentences_unstemmed[i + 1]if i + 1 < len(temp_sentences_unstemmed)else "")for i in range(0, len(temp_sentences_unstemmed), 2)]
    documents_lists.append(sub_doc)
    document = document.split()  # split the doc into words to prepare it for stemming
    stemmed_document = " ".join([stemmer.stem(word) for word in document])  # stem and join the text back
    stemmed_documents.append(stemmed_document)  # this produces a list of strings that we use for TF-IDF
    temp_sentences_stemmed = re.split("([.!?])\s+", stemmed_document)  # the same splitting with .!? as delimiters
    sub_stemmed_doc = [temp_sentences_stemmed[i]+ (temp_sentences_stemmed[i + 1] if i + 1 < len(temp_sentences_stemmed) else "")for i in range(0, len(temp_sentences_stemmed), 2)]
    stemmed_documents_lists.append(sub_stemmed_doc)  # this now produces a list of our docs, where each doc is a list of sentences. This is only for context.


# Boolean search
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


def boolean_return(user_query):
    hits_list = []

    hits_matrix = eval(rewrite_query(user_query))  # the query
    hits_list = list(hits_matrix.nonzero()[1])

    if len(hits_list) > 3:
        hits_list = hits_list[:3]

    result = []
    result.append(f"Query: {user_query}")
    for doc_idx in hits_list:
        doc_result = {}
        doc_result["Title"] = titles[doc_idx]
        doc_result["Link"] = httplinks[doc_idx]

        # Boolean match is exact match
        for j, sentence in enumerate(documents_lists[doc_idx]):
            query_words = set(user_query.split())
            sentence_words = set(sentence.split())

            if query_words.intersection(sentence_words):
                doc_result["Preview"] = documents_lists[doc_idx][j]
                break

        result.append(doc_result)

    formatted_results = result[0] + "<br/>"  # Make the results look a bit nicer
    for res in result[1:]:
        formatted_results += 'Title: {}<br/>'.format(res.get('Title', 'N/A')) 
        formatted_results += 'URL: {}<br/>'.format(res.get('Link', 'N/A'))
        formatted_results += 'Preview: {}<br/><br/>'.format(res.get('Preview', 'N/A'))
    return formatted_results


# TF-IDF search
tfidf = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
tf_idf_matrix = tfidf.fit_transform(stemmed_documents).T.tocsr()  # using stemmed docs!


def search_with_TFIDF(query_string, exact_match=False):
    # Vectorize query string
    query_vec = tfidf.transform([query_string]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, tf_idf_matrix)

    # Rank hits
    ranked_scores_and_doc_ids = sorted(
        zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True
    )

    results = []  # Initialize an empty list to store results
    results.append(
        f"Query: {query_string}"
    )  # Append the query to the results list as the first element

    seen_doc_indices = set()
    unique_docs_found = 0

    # Loop through ranked documents
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
        # Skip if we've already seen this document
        if doc_idx in seen_doc_indices:
            continue

        doc_result = {
            "title": titles[doc_idx],
            "score": round(score, 2),
            "url": httplinks[doc_idx],
            "preview": "",
        }

        # Check for sentence matches
        if not exact_match:  # if the query is not an exact match
            for j, stemmed_sentence in enumerate(stemmed_documents_lists[doc_idx]):
                if any(word in stemmed_sentence for word in query_string.split()):
                    doc_result["preview"] = documents_lists[doc_idx][j]
                    break  # Break after the first match to avoid printing multiple sentences from the same document
        else:  # if the query is an exact match
            for j, sentence in enumerate(documents_lists[doc_idx]):
                query_words = set(query_string.split())
                sentence_words = set(sentence.split())

                if query_words.intersection(sentence_words):
                    doc_result["preview"] = documents_lists[doc_idx][j]
                    break  # Break after the first match to avoid printing multiple sentences from the same document

        results.append(doc_result)  # Append the result dict to the results list

        seen_doc_indices.add(doc_idx)
        unique_docs_found += 1
        if unique_docs_found == 3:  # Stop after finding 3 unique documents
            break
    
    formatted_results = results[0] + "<br/>"  # Make the results look a bit nicer
    for res in results[1:]:
        formatted_results += 'Title: {}<br/>'.format(res['title'])
        formatted_results += 'Score: {}<br/>'.format(res['score'])
        formatted_results += 'URL: {}<br/>'.format(res['url'])
        formatted_results += 'Preview: {}<br/><br/>'.format(res['preview'])

    return formatted_results



# Sentence Bert
# model = SentenceTransformer("all-MiniLM-L6-v2")
# model.save(path="all-MiniLM-L6-v2")
# Load Sentence-BERT model
try:
    model = SentenceTransformer(path="all-MiniLM-L6-v2")
except:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save(path="all-MiniLM-L6-v2")

# Assuming `documents` is a list of your document texts
# Compute embeddings for all documents (ideally done once and cached)
document_embeddings = model.encode(documents_lists, convert_to_tensor=True)


def search_with_embeddings(query):
    # Encode the query to get its embedding
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # Rank documents based on the cosine similarity
    ranked_doc_indices = np.argsort(-cosine_scores.cpu().numpy())

    results = []  # Initialize an empty list to store results
    results.append(
        f"Query: {query}"
    )  # Append the query to the results list as the first element
    # Display top 3 similar documents, do not show the duplicates
    unique_docs_found = 0
    seen_doc_indices = set()
    for idx in ranked_doc_indices:
        if idx in seen_doc_indices:
            continue
        else:
            doc_result = {
                "title": titles[idx],
                "score": f"{cosine_scores[idx].item():.4f}",
                "url": httplinks[idx],
                "preview": f"{documents[idx][:150]}...",  # Display first 150 characters as a preview
            }
            results.append(doc_result)  # Append the result dict to the results list

            seen_doc_indices.add(idx)
            unique_docs_found += 1
            if unique_docs_found == 3:  # Stop after finding 3 unique documents
                break
    formatted_results = results[0] + "<br/>"  # Make the results look a bit nicer
    for res in results[1:]:
        formatted_results += 'Title: {}<br/>'.format(res['title'])
        formatted_results += 'Score: {}<br/>'.format(res['score'])
        formatted_results += 'URL: {}<br/>'.format(res['url'])
        formatted_results += 'Preview: {}<br/><br/>'.format(res['preview'])            
    return formatted_results


# QUERY


def function_query(bort, user_query):

    if bort == "b":
        return boolean_return(user_query)

    # using TF-IDF search
    elif bort == "t":

        if '"' in user_query:  # if the query contains quotes, skip the stemming
            # replace " with space
            user_query = user_query.replace('"', " ")
            search_with_TFIDF(user_query, exact_match=True)
        else:
            stemmed_query = " ".join(stemmer.stem(word) for word in user_query.split())
            print("stemmed query:", stemmed_query)
            try:
                return search_with_TFIDF(stemmed_query, exact_match=False)
            except:
                print("Invalid query, please try again.")

    # using fuzzy search
    elif bort == "s":

        return search_with_embeddings(user_query)
    # elif bort == "":
    #     break
    else:
        print("Invalid input. Please try again.")
        print()  # if the user hits enter, the program will exit


if __name__ == "__main__":
    function_query()
