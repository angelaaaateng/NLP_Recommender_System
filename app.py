

import gensim
from flask import Flask, request, render_template
import io
import flask
import json
from recommender_lib import generate_recommendations


app = Flask(__name__, template_folder='templates')
#for more information on flask easy startup see
#http://flask.pocoo.org/docs/1.0/quickstart/ ;
#https://teamtreehouse.com/community/can-someone-help-me-understand-flaskname-a-little-better
model = None

'''
Import Flask class and create an instance of this class
Note that __name__ is the name of the current class, function,
method, descriptor or generator instance.
'''


def load_model():
    '''
    Load the pre-trained word2vec model and define it as a global variable
    that we can use after startup
    '''
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    print("* Model loaded successfully")

def find_vocab():
    '''
    Returns the model vocuabulary.

    Output: (dict)
    '''
    return model.vocab
    print("* Model Vocab Loaded")

@app.route('/')
#def my_form():
#    return render_template('./templates/my-form.html')
def form():
    return("""
        <html>
            <body>
                <h1><center>NLP Recommender System</h1>

                <form action="/pangeaapp" method="POST" enctype="multipart/form-data">
                    <center> <input type="file" name="title" /> </center>
                    <br>
                    <p> </p>
                    <center> <input type="submit"/> </center>
                </form>



            </body>
        </html>
    """)

# @app.route('/', methods=['POST'])
# def some_function():
#     text = request.form.get('textbox')
#def my_form_post():
#    text = request.form['text']
#    processed_text = text.upper()
#    return processed_text
# <form method="POST">
#     <input name="text">
#     <input type="submit">
# </form>
# @app.route("/", methods=["GET, POST"])
@app.route("/pangeaapp", methods=['GET', 'POST'])
def predict():
    '''
    Specify the app route using a route decorater to
    bind the predict function to a url, and then check
    to ensure the json req was properly uploaded to endpoint.
    Use POST method to send HTML from data to server. Note that we used
    POST instead of GET since information is contained in the message body rather than the URL.

    Output: JSON with title and cosine similarity score (dict)
    '''
    print("* Requesting JSON data -- API")

    f = request.files['title']

    if not f:
        return("No file selected. Please choose a JSON file and try again.")

    data = {"success": False}
    print("* Initialization ok")
    if flask.request.method == "POST":
    #flask methods: https://www.tutorialspoint.com/flask/flask_http_methods.htm

        # title = request.form["title"]
        print("* Locating Title")
            # title = json.loads(flask.request.data)['f']
        title = request.get(f).json()
        # title = json.loads(flask.request.data)["title"]
        print("* Input Title: ", title)

            #using generate recommendations from pangea python script
        data["recommendations"] = generate_recommendations(title, model)
            # indicate that the request was a success
        data["success"] = True
    # ensure that a json request was properly uploaded to our endpoint
    # if flask.request.method == "POST":
    # #flask methods: https://www.tutorialspoint.com/flask/flask_http_methods.htm
    #     if flask.request.data:
    #     # title = request.form["title"]
    #         print("* Locating Title")
    #         # title = json.loads(flask.request.data)['f']
    #         title = json.loads(f)
    #         #title = json.loads(flask.request.data)["title"]
    #         #not using encode('ascii','ignore') as it throws an error
    #     # print("*Input Title: ", title)
    #         print("* Input Title: ")
    #         print(title)
    #         #using generate recommendations from pangea python script
    #         data["recommendations"] = generate_recommendations(title, model)
    #         # indicate that the request was a success
    #         data["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

    '''
    Checks to see if the name module was called interactively and
    then call the specified function to execute the code.
    '''
if __name__ == "__main__":
    print(("* Loading gensim model and Flask starting server..."
        "please wait until server has fully started"))
    app.debug = True
    #set app debug to true so that whenever a change is made on .py code,
    #it reflects on server/client terminal tools
    load_model()
    app.run()
