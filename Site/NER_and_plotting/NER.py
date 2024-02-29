# This thing NERs our medical document. For it to work you need the model downloaded to your computer. It also runs for a while.
import json

# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')

# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")


# import the documents
with open("Site/medical_document.txt", "r", encoding="utf-8") as f:
    articles = f.read()
articles = articles.split("\n\n")  # split by two new lines
articles = articles[-50:] # takes only last 50 articles
articles = str(articles)

# process in chunks so that it doesn't blow up because it needs too much memory
chunks = [articles[i:i + 1000] for i in range(0, len(articles), 1000)] 

results = []
for chunk in chunks:
    result = pipe(chunk)
    results.append(result)

# save in JSON for serialization
entities=[]
for result in results:
    for entity in result:
        entities.append(dict(entity, score=str(entity['score'])))
with open('Site/NER_and_plotting/NERed_recent_medical_documents.json', 'w') as f:
    json.dump(entities, f, indent=4)

