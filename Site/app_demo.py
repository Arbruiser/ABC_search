from flask import Flask, render_template, url_for
# Please import database

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test_index.html')

if  __name__ == "__main__":
    app.run(debug=True)