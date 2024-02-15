from flask import Flask, render_template, url_for, request

from search import function_query


app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("search_UI.html")


@app.route(
    "/search", methods=["POST"]
)  # Please complete the method, Baiyi will synchronize the layout of the return page.
def search():
    # Get the search query from the form
    query = request.form["query"]
    search_method = request.form[
        "search_method"
    ]  # This tells you which button was clicked

    # You can now use the search_method value to determine how to process the query
    if search_method == "Boolean Search":
        # Process the query using Boolean search
        results = f"Boolean search for '{query}'"
        function_query(bort="b", user_query=query)
    elif search_method == "TF-IDF Search":
        # Process the query using TFIDF
        results = f"TFIDF search for '{query}'"
        function_query(bort="t", user_query=query)
    elif search_method == "Fuzzy Search":
        # Process the query using Fuzzy search
        # results = f"Fuzzy search for '{query}'"
        results = function_query(bort="s", user_query=query)
    else:
        # Default case or error handling
        results = "Invalid search method selected."

    return results


if __name__ == "__main__":
    app.run(debug=True)
