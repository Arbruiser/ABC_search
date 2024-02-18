# plots a chart with the most common words.
import matplotlib.pyplot as plt
import json

with open("NERed_recent_medical_documents.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Categories that we will plot
categories = ['DISEASE_DISORDER', 'SIGN_SYMPTOM', 'MEDICATION', 'BIOLOGICAL_STRUCTURE']

# Separate results for different categories
filtered_results_categories = {category: {} for category in categories} # makes a list where keys are the categories and 

for result in results:
    if float(result['score']) > 0.5: # only add word if the model is relatively confident
        key = result['word'] 
        if result['entity_group'] in categories:
            if key in filtered_results_categories[result['entity_group']]:
                filtered_results_categories[result['entity_group']][key] += 1
            else:
                filtered_results_categories[result['entity_group']][key] = 1

for category in categories:
    # Get the n most common words and their frequencies
    most_common_filtered = {k: v for k, v in sorted(filtered_results_categories[category].items(), key=lambda item: item[1], reverse=True)[:10]}

    # Create bar plot
    plt.figure(figsize=(10,5))
    plt.bar(most_common_filtered.keys(), most_common_filtered.values())
    plt.title(f'Entity Frequency for {category}')
    plt.xlabel('Entities')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'Plots/trending_{category}.png')