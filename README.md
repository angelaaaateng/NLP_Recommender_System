# Pangea Recommender System


## Recommender System - Marketplace Matching Script for Pangea.App


- **Recommender_lib.py** - Python Script containing Word2Vec model and recommender system
- **app.py** - flask application for deploying the recommender system; Flask App for Pangea.App
- **requirements.txt** - requirements needed to run the model
- **Pangea_RS_Cleaned.ipnyb** - Jupyter Notebook that outlines word2vec model and data output line by line
- **App_Notebook_Viz.ipynb** - Jupyter notebook with more detailed explanation of each part of the code.


## Notes for running flask app:
- remember to check both the client and server side
- use the curl command to call on the flask app:
curl localhost:5000/pangeaapp -d '{"title": "Teach me how to cook!"}' -H 'Content-Type: application/json'

## Transfer Learning:
- Download the Google Vectors Bin file here https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
