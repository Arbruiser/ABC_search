from flask import Flask, render_template, url_for, request
# Please import database

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('frontpage.html')

@app.route('/') # Please complete the method, Baiyi will synchronize the layout of the return page.
def index():
    return render_template('returnpage.html')

if  __name__ == "__main__":
    app.run(debug=True)