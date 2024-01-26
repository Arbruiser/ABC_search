#!/usr/bin/env python3

# NEED TO ADD OUR DOCS INSTEAD
with open('sample_static.txt', 'r') as f:
    content = f.read()

documents = content.split('\n\n') # makes a list of our docs

# Need to have sklearn installed
from sklearn.feature_extraction.text import CountVectorizer


# Make a matrix of our terms and convert it to dense
cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(documents)
dense_matrix = sparse_matrix.todense()

td_matrix = dense_matrix.T   # .T transposes the matrix

terms = cv.get_feature_names_out()

#print("\nterm -> IDX mapping:\n") # for testing
#print(cv.vocabulary_) # note the _ at the end


# SEARCHING
t2i = cv.vocabulary_  # shorter notation: t2i = term-to-index
#print("Query: example")
#print(td_matrix[t2i["example"]])


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

def run_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query))) # Eval runs the string as a Python command
    print()



# SCALING UP TO LARGER DOC COLLECTIONS
#print(sparse_matrix.tocsc()) # to check different matrices
#print(sparse_matrix.T)
sparse_td_matrix = sparse_matrix.T.tocsr() #makes the matrix ordered by terms, not documents
#print(sparse_td_matrix)

def rewrite_token(t):
    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t)) # Make retrieved rows dense


while True:
	user_query = input("Make a query with operands:")
	if user_query=="":
		break
	run_query(str(user_query))
	# SHOW RETRIEVED DOCUMENTS
	hits_matrix = eval(rewrite_query(user_query))
	print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
	print("The coordinates of the non-zero elements:", hits_matrix.nonzero())

	hits_list = list(hits_matrix.nonzero()[1])
	print(hits_list)

	for doc_idx in hits_list:
		print("Matching doc title:", documents[doc_idx])

	for i, doc_idx in enumerate(hits_list):
    		print("Matching doc #{:d}: {:s}".format(i, documents[doc_idx]))
