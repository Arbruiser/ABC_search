# ABC_search
Our insanely cool group working on a search engine in medical field.

This search engine is able to perform Boolean search (for higher precison) or TF-IDF search (for higher recall), retrieving   medical text data from [MedicalNewsToday](https://www.medicalnewstoday.com/).

Besides the search program, the project also consists of a scraper, which performs automatic scrapping on [MedicalNewsToday](https://www.medicalnewstoday.com/).

Contributed by Artur, Baiyi and Chao.

**Use instruction:**

1. Git clone "git clone https://github.com/Arbruiser/BACoN.git"
2. In the venv run "pip3 install ." to install all the dependencies.

**Major updates:**
Update: 2024-02-15
1. Integrate the search with flask so we have a web interface.

Update: 2024-02-08
1. Added the fuzzy search using sentence-transformers. Now the user can choose between Boolean, TF-IDF, and fuzzy search.
2. When user's query contains quotes, the program will not stem the word in the quotes. 
3. Added a handy tool "de_duplicate.ipynb"  to remove duplicate data.

Bug fixed:
The top 3 results now will be unique.

Update: 2024-02-07
1. Integrate the query function, now the query will ask user to choose between Boolean and TF-IDF search.
2. Added colorama to highlight the context of the query (only on boolean search result for testing).

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

--- If the query contains multiple words, the context function will look for ANY of the query words in the text in order and output only the first 3 matches. Try running "sneezing is not an illness". 

Work before the class:
--- Let user choose if to use Boolean or TF-IDF search. Maybe with the first input asking for B (Boolean) or T (TF-IDF)?  

--- Can implement indexing of the sentences in the matched document to output the highest scoring sentences first. 

--- Can implement highlighting the matched word in the sentence in the same fashion as looking for unstemmed docs. We turn each sentence into
stemmed and unstemmed lists of words. Search for the stemmed query word in the stemmed lists and put *...* around the word from the same number of document and the same number of sentence and the same number of word 

Update: 2024-02-01
1. Now the search result only returns the first 3 matches.
2. Added more documents to the corpus. Now is 13.

Update: 2024-01-29

1. Now the search result includes the document's title, link, and the context of the query.
The result is in below format:

\<Document Title\>

\<Document Link\>

... \<The sentence includes the query keyword\> ...

Note: The context sentence only include the first keyword in the query.

2. Now if the query is invalid, the program will not crash. Instead, it will print "Invalid query, please try again." and ask for another query.

Minor changes:

1. Cleaned up the code a little bit.
2. Removed the unnecessary variables and functions.

Future work:

~- Will add more documents to the corpus.~
~- Try to rank the documents (maybe by word frequency or other methods). Done~
~- Make logic that will allow searching for multiple words with some unknown words (Arthur)~

~Challenges:~
~- If the query has more than one keyword, try to show the sentence that includes all the keywords. Done~

~- User doesn't have to type "and" or "AND" to use the AND operand. They can just type a space between two keywords. Done~

~- Using embedding models to do fuzzy search.~

