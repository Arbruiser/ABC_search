# BACoN
Our insanely cool group working on a search engine in medical field.

This search engine is able to perform Boolean search (for higher precison) or TF-IDF search (for higher recall), retrieving   medical text data from [MedicalNewsToday](https://www.medicalnewstoday.com/).

Besides the search program, the project also consists of a scraper, which performs automatic scrapping on [MedicalNewsToday](https://www.medicalnewstoday.com/).

Major updates:

Update: 2024-02-07
1. Integrate the query function, now the query will ask user to choose between Boolean and TF-IDF search.
2. Added colorama to highlight the context of the query (only on boolean search result for testing).

Update: 2024-02-03
(Arthur) Added Porter stemmer. Now we use stemmed documents for making our matrices. User query is also stemmed. 

---Now we also show the score of the matching doc. We can translate it into plain English with a bunch of if statements
like "if score>0.3 then "score very high!" and so on.

--- If the query contains multiple words, the context function will look for ANY of the query words in the text in order 
and output only the first 3 matches. Try running "sneezing is not an illness". 

-- Now the scraper can scrape new text data automatically.

Contributed by Artur, Baiyi and Chao.