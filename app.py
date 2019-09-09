

import gensim
from flask import Flask, request, render_template
import io
import flask
import json
from recommender_lib import generate_recommendations
import numpy as np
import pandas as pd
import imgkit

import matplotlib.pyplot as plt
from io import StringIO

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
    #model = {"cook": np.array([0.5, 0.6])}
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
    # return("""
    #     <html>
    #         <body>
    #             <h1><center>NLP Recommender System</h1>
    #             <form action="/pangeaapp" method="POST" enctype="multipart/form-data">
    #                 <center> <input type="text" name="title" /> </center>
    #                 <br>
    #                 <p> </p>
    #                 <center> <input type="submit"/> </center>
    #             </form>
    #
    #
    #
    #         </body>
    #     </html>
    # """)

    return("""
        <html>
            <body>
              <form action = "http://localhost:5000/pangeaapp" method = "post">
                 <p>Enter Title:</p>
                 <p><input type = "text" name = "title" /></p>
                 <p><input type = "submit" value = "submit" /></p>
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
    print(request.form['title'])
    if not request.form or not request.form['title']:
        print("!! Title not read !! ")
    title =  request.form['title']
        #'done': False
    #}
    #title.append(title)

    #f = request.files['title']
    #title = request.get_json(['title'])
    print(type(title))
    # f = request.files['title']

    #if not title:
    #    return("No file selected. Please choose a JSON file and try again.")

    data = {}
    #data = {"success": False}
    print("* Initialization ok")
    if flask.request.method == "POST":
    #flask methods: https://www.tutorialspoint.com/flask/flask_http_methods.htm

        # title = request.form["title"]
        print("* Locating Title")
            # title = json.loads(flask.request.data)['f']
        #print(type(request.args))
        #title = request.args.get(f).json()
        # title = json.loads(flask.request.data)["title"]
        print("* Input Title: ", title)

            #using generate recommendations from pangea python script
        data["recommendations"] = generate_recommendations(title, model)
            # indicate that the request was a success
        #data["success"] = True

    # css = """
    #         <style type=\"text/css\">
    #         table {
    #         color: #333;
    #         font-family: Helvetica, Arial, sans-serif;
    #         width: 640px;
    #         border-collapse:
    #         collapse;
    #         border-spacing: 0;
    #         }
    #
    #         td, th {
    #         border: 1px solid transparent; /* No more visible border */
    #         height: 30px;
    #         }
    #
    #         th {
    #         background: #DFDFDF; /* Darken header a bit */
    #         font-weight: bold;
    #         }
    #
    #         td {
    #         background: #FAFAFA;
    #         text-align: center;
    #         }
    #
    #         table tr:nth-child(odd) td{
    #         background-color: white;
    #         }
    #         </style>
    # """

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
    #return data["recommendations"][0] #flask.jsonify({'task': task}),

        #save output as json
        print("Data", data)
        print("***END OF DATA***")
        print("First Entry:", data['recommendations'][0])
        print("First 10 Entryies:", data['recommendations'][0:10])

        final_output = data['recommendations'][0:10]
        print("TYPE", type(final_output))
        #fig, ax = plt.subplot(111, frame_on=False) # no visible frame
        #ax.xaxis.set_visible(False)  # hide the x axis
        #ax.yaxis.set_visible(False)  # hide the y axis

        #ax.table(ax, final_output)  # where df is your data frame

        #plt.savefig('./Static/reco_image.png')
        #data_json = flask.jsonify(data)

        #turn json into dataframe
        #final_output = pd.read_json(data_json)
        print(final_output)

        #final_output = data

        #save dataframe as image... lol
        #reco_image = final_output.write(data.to_html())

        #write css
        # final_output.write(css)

        # write HTML-ized pandas df
        final_output = StringIO()
        final_output.write(data.to_html())
        final_output.close()

        #crop final image
        print("* Saving results in an image....")
        imgkitoptions = {"format": "png"}
        imgkit.from_file("filename.html", outputfile, options=imgkitoptions)


    return render_template("recommendations.html", reco_image = './Static/reco_image.png')

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
