

import gensim
from flask import Flask
import io
import flask
import json
from recommender_lib import generate_recommendations


app = Flask(__name__)
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


@app.route("/pangeaapp", methods=["POST"])
def predict():
    '''
    Specify the app route using a route decorater to
    bind the predict function to a url, and then check
    to ensure the json req was properly uploaded to endpoint.
    Use POST method to send HTML from data to server. Note that we used
    POST instead of GET since information is contained in the message body rather than the URL.

    Output: JSON with title and cosine similarity score (dict)
    '''
    data = {"success": False}
    print("* Initialization ok")
    # ensure that a json request was properly uploaded to our endpoint
    if flask.request.method == "POST":
    #flask methods: https://www.tutorialspoint.com/flask/flask_http_methods.htm
        if flask.request.data:
            title = json.loads(flask.request.data)["title"]
            #not using encode('ascii','ignore') as it throws an error
            print("*Input Title: ")
            print(title)
            #using generate recommendations from pangea python script
            data["recommendations"] = generate_recommendations(title, model)
            # indicate that the request was a success
            data["success"] = True
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
