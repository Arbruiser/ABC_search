#!/usr/bin/env python3
"""
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
- Try to rank the documents (maybe by word frequency or other methods).

Challenges:
- If the query has more than one keyword, try to show the sentence that includes all the keywords.
- User doesn't have to type "and" or "AND" to use the AND operand. They can just type a space between two keywords.
- Using embedding models to do fuzzy search.

"""
# import dependencies
from sklearn.feature_extraction.text import CountVectorizer
import re

# import the documents
with open("medical_document.txt", "r") as f:
    content = f.read()

documents = content.split("\n\n")  # makes a list of our docs

# the https links are the first line of each doc before the first \n
# the titles are the second line of each doc before the second \n

# Get the first line and second line of each doc
httplinks = []
titles = []

for i in range(len(documents)):
    httplinks.append(documents[i].split("\n")[0])
    titles.append(documents[i].split("\n")[1])

# Remove the httplinks and titles from the documents
for i in range(len(documents)):
    documents[i] = documents[i].replace(httplinks[i], "")
    documents[i] = documents[i].replace(titles[i], "")


# Make a matrix of our terms and convert it to dense
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


# QUERY
while True:
    user_query = input(
        "Hit enter to exit. Your query to search: "
    )
    if user_query == "":
        break

    # SHOW RETRIEVED DOCUMENTS
    try:
        print("Query: '" + user_query + "'")
        hits_matrix = eval(rewrite_query(user_query))  # the query
        print("rewritten query:", rewrite_query(user_query)) # for debugging

        hits_list = list(hits_matrix.nonzero()[1])
        # print(hits_list)

        # if there are more than 3 matches, limits the showed matches to 3
        if len(hits_list)>3:
            hits_list=hits_list[:3]

        for doc_idx in hits_list:
            print()
            print(titles[doc_idx])  # print the title
            print(httplinks[doc_idx])  # print the link

            if len(user_query.split()) == 1:
                # regex to find the sentence with the query
                pattern = r"([^.!?]*" + user_query + r"[^.!?]*[.!?])"
                match = re.search(pattern, documents[doc_idx], re.IGNORECASE)
                print("... " + match.group(0) + " ...")
                print()
            else:  # only show the context of the first term in the query
                pattern = r"([^.!?]*" + user_query.split()[0] + r"[^.!?]*[.!?])"
                match = re.search(pattern, documents[doc_idx], re.IGNORECASE)
                print("... " + match.group(0) + " ...")
                print()

    except:
        print("Invalid query, please try again.")
        print()
