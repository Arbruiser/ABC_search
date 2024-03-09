from flask import Flask, render_template, url_for, request, redirect
from search import function_query
import random
import os
import time
app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("search_UI.html")


@app.route("/search", methods=["POST"])
def search_redirect():
    search_method = request.form["search_method"]
    query = request.form["query"]

    return redirect(url_for("search", search_method=search_method, query=query))


@app.route("/search/<search_method>/<query>", methods=["GET"])
def search(query, search_method):
    # You can now use the search_method value to determine how to process the query
    if search_method == "Boolean Search":
        # Process the query using Boolean search
        start_time = time.time()
        results,result_num= function_query(bort="b", user_query=query)
        end_time = time.time()

    elif search_method == "TF-IDF Search":
        # Process the query using TFIDF
        start_time = time.time()
        results,result_num = function_query(bort="t", user_query=query)
        end_time = time.time()

    elif search_method == "Semantic Search":
        # Process the query using Fuzzy search
        # results = f"Fuzzy search for '{query}'"
        start_time = time.time()
        results,result_num = function_query(bort="s", user_query=query)
        end_time = time.time()
        
    if results:
        # return results
        time_taken = round(end_time - start_time,2)
        return render_template("return.html", results=results,result_num=result_num,time_taken=time_taken)
    else:
        # Get the path to the folder containing the funny animal images
        folder_path = "Site/static/css/Crash_animals"

        # Get a list of all the image files in the folder
        funny_animals = [
            file
            for file in os.listdir(folder_path)
            if file.endswith((".jpg", ".jpeg", ".png", "avif"))
        ]

        # Select a random image from the list
        selected_image = random.choice(funny_animals)

        # Rest of the code remains the same
        random_number = random.randint(1, 100)
        if random_number == 42:
            selected_image = "RARE/rare_lucky_golden_cat.jpg"
        return render_template("crashpage.html", animal_image=selected_image)


@app.route("/trending")
def trending():
    return render_template("trending.html")


if __name__ == "__main__":
    app.run(debug=True)
