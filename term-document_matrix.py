#!/usr/bin/env python3

# NEED TO ADD OUR DOCS INSTEAD
documents = ["This is a silly example",
             "A better example",
             "Nothing to see here",
             "This is a great and long example"]

# Need to have sklearn installed
from sklearn.feature_extraction.text import CountVectorizer


# Make a matrix of our terms
cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(documents)
#print("Term-document matrix: (?)\n")
#print(sparse_matrix)


dense_matrix = sparse_matrix.todense()
#print("Term-document matrix: (?)\n")
#print(dense_matrix)


td_matrix = dense_matrix.T   # .T transposes the matrix
#print("Term-document matrix:\n")
#print(td_matrix)


#print("\nIDX -> terms mapping:\n")
#print(cv.get_feature_names_out()) # prints the words in the order they are in the matrix


terms = cv.get_feature_names_out()
#print("First term (with row index 0):", terms[0])
#print("Third term (with row index 2):", terms[2])


print("\nterm -> IDX mapping:\n")
print(cv.vocabulary_) # note the _ at the end


#print("Row index of 'example':", cv.vocabulary_["example"]) # can be used for testing, but we should add input() from the user



# SEARCHING
t2i = cv.vocabulary_  # shorter notation: t2i = term-to-index
print("Query: example")
print(td_matrix[t2i["example"]])


# QUERY PARSING
# Operators and/AND, or/OR, not/NOT become &, |, 1 -
# Parentheses are left untouched
# Everything else is interpreted as a term and fed through td_matrix[t2i["..."]]

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements

def rewrite_token(t):
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t)) # when we detect and/or/not we replace them with our operands

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query))) # Eval runs the string as a Python command
    print()



# SCALING UP TO LARGER DOC COLLECTIONS
#print(sparse_matrix.tocsc())
#print(sparse_matrix.T)
sparse_td_matrix = sparse_matrix.T.tocsr() #makes the matrix ordered by terms, not documents
#print(sparse_td_matrix)


def rewrite_token(t):
    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t)) # Make retrieved rows dense

test_query("NOT example OR great")




# SHOW RETRIEVED DOCUMENTS
hits_matrix = eval(rewrite_query("NOT example OR great"))
print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
print("The coordinates of the non-zero elements:", hits_matrix.nonzero())

hits_list = list(hits_matrix.nonzero()[1])
print(hits_list)

for doc_idx in hits_list:
    print("Matching doc:", documents[doc_idx])

for i, doc_idx in enumerate(hits_list):
    print("Matching doc #{:d}: {:s}".format(i, documents[doc_idx]))
