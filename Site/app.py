from flask import Flask, render_template, url_for, request, redirect
from search import function_query
import random

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

        results = function_query(bort="b", user_query=query)
    elif search_method == "TF-IDF Search":
        # Process the query using TFIDF

        results = function_query(bort="t", user_query=query)
    elif search_method == "Fuzzy Search":
        # Process the query using Fuzzy search
        # results = f"Fuzzy search for '{query}'"
        results = function_query(bort="s", user_query=query)
    else:  # apparently this does nothing now and can be deleted
        # Default case or error handling
        results = "Invalid search method selected."

    if results:
        # return results
        return render_template("return.html", results=results)
    else:
        funny_animals = ['funny_chameleon.jpg', 'funny_chihuahua.jpg', 'funny_dog.jpg', 'funny_dogchicken.jpg', 'funny_fish.jpg', 'funny_llama.jpg', 'funny_meercat.jpg', 'funny_penguins.jpg', 'funny_seals.jpg', 'funny_sloth.jpg', 'funny_squirrel.jpeg', 'seductive_chimp.jpg', 'stuck_dog.jpg', 'cozy_chihuahua.jpg', 'crash_cat.png', 'danger_cats.jpg', 'funny_bear.jpg',     "also_her.jpg", "bad_worse.jpg",  "bober.jpg", "casowary.jpg", "fish.jpg", "fox.jpg", "headless_penguin.jpg", "heart_attack.jpg", "huge_eyes.jpg", "human_wolf.jpg",
        "into_parrot.jpg", "look_natural.jpg", "lose_weight_cat.jpg", "loved_seal.jpg", "mean_me.jpg", "monkey_mum.jpg", "moose.jpg", "no_taxes.jpg", "owl.jpg", "panda_meme.jpg", "parents_meme.jpg", "patient_pigeons.jpg", "places_to_be.jpg", "positive_possum.jpg", "puppy.jpg", "raccoon.jpg", "sealed_door.jpg", "smiley_fish.jpg", "smiley_owl.jpg", "someone_hey.jpg", "squirrel.jpg", "start_drama.jpg", "stronger_rat.jpg", "the_hard_way.jpg", "tiger.jpg", "touches_foot.jpg", "winged_cow.jpg", "winter_dog.jpg", "worried.jpg", "money.jpg"]
        selected_image = random.choice(funny_animals)
        random_number = random.randint(0, 100)
        if random_number==42:
            selected_image="rare_lucky_golden_cat.jpg"
        return render_template("crashpage.html", animal_image=selected_image)


@app.route("/trending")
def trending():
    return render_template("trending.html")


if __name__ == "__main__":
    app.run(debug=True)
