# ABC_search
This search engine is able to perform Boolean search, TF-IDF search or semantic search with sBERT, retrieving medical text data from [MedicalNewsToday](https://www.medicalnewstoday.com/).  

Besides the search engine, our project also contains a scraper, which performs automatic scrapping on [MedicalNewsToday](https://www.medicalnewstoday.com/) and medical NER that we use for plotting.  
All the code also contains comments about what it does, so you can easily find out how how exactly everything works if you wish so.  

Contributed by Artur, Baiyi and Chao (ABC are our initials).  

**Use instructions:**
1. Git clone "git clone https://github.com/Arbruiser/ABC_search.git".
2. In the venv run "pip3 install ." to install all the necessary dependencies automatically.
3. Run the command ./run_site from the root of the repository to automatically get the website running.
4. Open http://127.0.0.1:8000 in your browser.
  
## Description of files:
### **ABC_search/** :   
_de_duplicate.ipynb_ -  raw text processing

_medical_document.txt_ - our collection of scraped articles which are separated by double new lines. Each article consists of link to the original, the title and the body of the article. Each sentence is its own line.  
_run_site.sh_ - automatically runs the website in debugging mode.   
_scraper.ipynb_ - scrapes the aforementioned website which publishes medical articles. 
_setup.py_ - Stores all the necessary dependencies for you to easily install.    
    
    
### **ABC_search/Site** :
_app.py_ - Flask script that runs the website.
_search.py_ - the main search algorithm with Boolean, TF-IDF and semantic searches. 
  
### **ABC_search/Site/all-MiniLM-L6-v2** :
Our sBERT model  
  
### **ABC_search/Site/NER_and_plotting** :
_NER.py_ - runs Medical NER from HuggingFace. [Clinical-AI-Apollo/Medical-NER](https://huggingface.co/Clinical-AI-Apollo/Medical-NER). It runs over 50 latest articles and stores the output in _NERed_recent_medical_documents.json_  
_NERed_recent_medical_documents.json_ - serialised output of _NER.py_ used for plotting.  
_plotting_regular.py_ - plots the frequency over our recent medical documents excluding stop words. These plots are currently not used. All produced plots are stored in /static/Plots/.     
_plotting_regular_NER.py_ - plots overall NER entity (not separate categories) frequency from the json file.  
_plotting_trending.py_ - plots NER entity frequency for 4 categories: "disease disorder", "sign symptom", "medication" and "biological structure".  


### **ABC_search/Site/static** :
Contains our CSS, background and images of funny animals that are used when the query doesn't match anything.  

### **ABC_search/Site/templates** :
Contains html files.       
     
     



