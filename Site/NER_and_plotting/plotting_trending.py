# plots a chart with the most common words.
import matplotlib.pyplot as plt
import json

with open("Site/NER_and_plotting/NERed_recent_medical_documents.json", "r", encoding="utf-8") as f:
    results = json.load(f)


# Categories that we will plot
categories = ["DISEASE_DISORDER", "SIGN_SYMPTOM", "MEDICATION", "BIOLOGICAL_STRUCTURE"]

# Separate results for different categories
filtered_results_categories = {category: {} for category in categories}  # makes a list where keys are the categories and

for result in results:
    if (float(result["score"]) > 0.5):  # only add word if the model is relatively confident
        key = result["word"]
        if result["entity_group"] in categories:
            if key in filtered_results_categories[result["entity_group"]]:
                filtered_results_categories[result["entity_group"]][key] += 1
            else:
                filtered_results_categories[result["entity_group"]][key] = 1

for category in categories:
    # Get the n most common words and their frequencies
    most_common_filtered = {k: v for k, v in sorted(filtered_results_categories[category].items(), key=lambda item: item[1], reverse=True)[:10]}
    formatted_category = category.lower().replace("_", " ").capitalize()  # lowercases the NER categories and replaces underscore with space
    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(most_common_filtered.keys(), most_common_filtered.values(), color="darkorchid")
    plt.title(formatted_category)
    plt.grid(axis="y", linestyle="-", linewidth=0.2)  # Adds thin horizontal gridlines
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f"Site/static/Plots/trending_{category}_horizontal.png")

    # vertical bar plots
    plt.figure(figsize=(10, 8))
    plt.barh(list(most_common_filtered.keys()), list(most_common_filtered.values()), color="darkorchid")
    plt.title(formatted_category)
    plt.grid(axis="x", linestyle="-", linewidth=0.2, alpha=0.6)  # Adds thin horizontal gridlines, zorder - grid lines don't appear on top of bars
    plt.ylabel("Frequency")
    plt.xticks(ha="right", fontsize=12)
    plt.gca().invert_yaxis()  # Inverts the order
    plt.subplots_adjust(left=0.25)
    plt.savefig(f"Site/static/Plots/trending_{category}_vertical.png")
