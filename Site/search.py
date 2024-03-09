#!/usr/bin/env python3

# import dependencies
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import torch




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
    documents[i] = (documents[i].replace(httplinks[i], "").replace("\n", " ").strip())  # replace remaining newline characters with space.
    documents[i] = (documents[i].replace(titles[i], "").replace("\n", " ").strip())  # replace remaining newline characters with space.


# Segment the documents into sentences
# Loop where we split the documents into sentences and stem. Brace yourselves.
stemmed_documents_lists = []  # list of lists
stemmed_documents = []  # list of strings
documents_lists = []  # again list of lists

for document in documents:
    temp_sentences_unstemmed = re.split("([.!?])\s+", document)  # here we split the sentences with delimiters being their own elements in the list. However, that's unnecessary now because in our doc each line is a sentence. We don't know if it stays that way, so let's keep it.
    # next line joins the delimiters with the previous sentence (don't worry about it)
    sub_doc = [temp_sentences_unstemmed[i]+ (temp_sentences_unstemmed[i + 1] if i + 1 < len(temp_sentences_unstemmed) else "") for i in range(0, len(temp_sentences_unstemmed), 2)]
    documents_lists.append(sub_doc)
    document = word_tokenize(document)    # split the doc into words to prepare it for stemming 
    stemmed_document = " ".join([stemmer.stem(word) for word in document])  # stem and join the text back
    stemmed_documents.append(stemmed_document)  # this produces a list of strings that we use for TF-IDF
    temp_sentences_stemmed = re.split("([.!?])\s+", stemmed_document)  # the same splitting with .!? as delimiters
    sub_stemmed_doc = [temp_sentences_stemmed[i] + (temp_sentences_stemmed[i + 1] if i + 1 < len(temp_sentences_stemmed) else "") for i in range(0, len(temp_sentences_stemmed), 2)]
    stemmed_documents_lists.append(sub_stemmed_doc)  # this now produces a list of our docs, where each doc is a list of sentences. This is only for context.


# Boolean search
# The boolean search will prioritize the titles
# Make a boolean matrix of our terms and convert it to dense
cv_titles = CountVectorizer(lowercase=True, binary=True)
sparse_matrix_titles = cv_titles.fit_transform(titles) 
dense_matrix_titles = sparse_matrix_titles.todense()
td_matrix_titles = dense_matrix_titles.T
t2i_titles = cv_titles.vocabulary_  # The dictionary for query

# if the query is not in the titles, it will search in the documents
cv_documents = CountVectorizer(lowercase=True, binary=True)
sparse_matrix_documents = cv_documents.fit_transform(documents)
dense_matrix_documents = sparse_matrix_documents.todense()
td_matrix_documents = dense_matrix_documents.T  # .T transposes the matrix
t2i_documents = cv_documents.vocabulary_ # The dictionary for query


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
    # Attempt to find the term in titles or documents
    term_in_titles = f'td_matrix_titles[t2i_titles.get("{t}", -1)]' if t in t2i_titles else None
    term_in_documents = f'td_matrix_documents[t2i_documents.get("{t}", -1)]' if t in t2i_documents else None
    
    if term_in_titles:
        return term_in_titles
    elif term_in_documents:
        return term_in_documents
    else:
        # Handle case where term is not found in either titles or documents
        return "np.zeros_like(td_matrix_documents[0])"  # Return a zero matrix of the same shape


def rewrite_query(query):  # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

sparse_td_matrix_titles = (sparse_matrix_titles.T.tocsr())
sparse_td_matrix_documents = (sparse_matrix_documents.T.tocsr()) 


def boolean_return(user_query):
    try:
        hits_matrix = eval(rewrite_query(user_query))  # the query
        hits_list = list(hits_matrix.nonzero()[1])

        result = [f"Query: {user_query}"]
        for doc_idx in hits_list[:99]:
            doc_result = {
                "title": titles[doc_idx],
                "url": httplinks[doc_idx],
                "score": "N/A",
                "preview": documents[doc_idx][:150] + "...",
            }
            result.append(doc_result)
        return result, len(hits_list)
    except:
        return [None,None]


# TF-IDF search
tfidf = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
tf_idf_matrix = tfidf.fit_transform(stemmed_documents).T.tocsr()  # using stemmed docs!


def search_with_TFIDF(user_query, query_string, exact_match=False):
    # Vectorize query string
    try:
        query_vec = tfidf.transform([query_string]).tocsc()
        # Cosine similarity
        # if np.dot(query_vec, tf_idf_matrix):
        hits = np.dot(query_vec, tf_idf_matrix)
        # Rank hits
        ranked_scores_and_doc_ids = sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
        # print(ranked_scores_and_doc_ids)
        results = []  # Initialize an empty list to store results
        results.append(f"Query: {user_query}")  # Append the query to the results list as the first element

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
            # print(doc_result)
            # Check for sentence matches
            if not exact_match:  # if the query is not an exact match
                for j, stemmed_sentence in enumerate(stemmed_documents_lists[doc_idx]):
                    if any(word in stemmed_sentence for word in query_string.split()):
                        try:
                            doc_result["preview"] = documents_lists[doc_idx][j]
                            break  # Break after the first match to avoid printing multiple sentences from the same document
                        except:
                            pass
            else:  # if the query is an exact match
                for j, sentence in enumerate(documents_lists[doc_idx]):
                    query_words = set(query_string.split())
                    sentence_words = set(sentence.split())

                    if query_words.intersection(sentence_words):
                        doc_result["preview"] = documents_lists[doc_idx][j]
                        break  # Break after the first match to avoid printing multiple sentences from the same document

            results.append(doc_result)  # Append the result dict to the results list
            # print(results)
            seen_doc_indices.add(doc_idx)
            unique_docs_found += 1
            if unique_docs_found == 99:  # Stop after finding 99 unique documents
                break

        return [results, unique_docs_found]
    except:
        return [None,None]


# Sentence Bert
# model = SentenceTransformer("all-MiniLM-L6-v2")
# model.save(path="all-MiniLM-L6-v2")
# Load Sentence-BERT model
model = SentenceTransformer("Site/all-MiniLM-L6-v2")

# Flatten the list of lists into a single list of sentences
all_sentences = [sentence for document in documents_lists for sentence in document]

# Compute embeddings for all sentences
# try to load the embeddings from a file if it exists to save time
file_path = "Site/all-MiniLM-L6-v2/sentence_embeddings.pt"

if os.path.exists(file_path):
    sentence_embeddings = torch.load(file_path)
else: # if the file doesn't exist, then compute
    sentence_embeddings = model.encode(all_sentences, convert_to_tensor=True)
    torch.save(sentence_embeddings, "Site/all-MiniLM-L6-v2/sentence_embeddings.pt")

# Create a mapping of sentences to their parent document index    
sentence_to_doc_index = []
for doc_index, document in enumerate(documents_lists):
    for sentence in document:
        sentence_to_doc_index.append(doc_index)

def search_with_embeddings(query):
    # Encode the query to get its embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
    # print(f"query_embedding shape: {query_embedding.shape}")
    # print(f"sentence_embeddings shape: {sentence_embeddings.shape}")
    # Compute cosine similarities

    cosine_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    # Rank sentences based on the cosine similarity
    ranked_sentences_indices = np.argsort(-cosine_scores.cpu().numpy())

    results = []  # Initialize an empty list to store results
    results.append(f"Query: {query}")  # Append the query to the results list as the first element
    # Display top 3 similar documents, do not show the duplicates
    unique_docs_found = 0
    seen_doc_indices = set()
    for idx in ranked_sentences_indices:
        doc_index = sentence_to_doc_index[idx]
        if doc_index in seen_doc_indices:
            continue
        # if the matched sentence is too short, skip it
        if len(all_sentences[idx]) < 25:
            continue
        elif cosine_scores[idx].item() < 0.25:  # Stop searching when the score is below 0.2
            break
        else:
            doc_result = {
                "title": titles[doc_index],
                "score": f"{cosine_scores[idx].item():.2f}",
                "url": httplinks[doc_index],
                "preview": f"{all_sentences[idx]}",  # Display first 150 characters as a preview
            }
            results.append(doc_result)  # Append the result dict to the results list

            seen_doc_indices.add(doc_index)
            unique_docs_found += 1
            if unique_docs_found == 99:  # Stop after finding 99 unique documents
                break

    return results,unique_docs_found


# QUERY


def function_query(bort, user_query):

    if bort == "b":
        
        return boolean_return(user_query)

    # using TF-IDF search
    elif bort == "t":
        if '"' in user_query or "'" in user_query:    # if the query contains quotes, skip the stemming
            # replace " with space
            quoted_query = user_query
            user_query = user_query.replace('"', "")
            stemmed_query = user_query
            return search_with_TFIDF(quoted_query, stemmed_query, exact_match=True)
        else:
            stemmed_query = " ".join(stemmer.stem(word) for word in user_query.split())
            print(stemmed_query)
            return search_with_TFIDF(user_query, stemmed_query, exact_match=False)

    # using fuzzy search
    elif bort == "s":

        return search_with_embeddings(user_query)
    # elif bort == "":
    #     break
    else:
        pass


if __name__ == "__main__":

    function_query()
