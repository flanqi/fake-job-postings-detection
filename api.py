import logging

import flask
import pickle
from flask_restful import reqparse
from flask import Flask, request, jsonify, Response
from src.cnn import pad_sequences
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

from src.preprocess import imbalance_correct

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NOTE this import needs to happen after the logger is configured


# Initialize the Flask application
application = Flask(__name__)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('comment_path')


def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


@application.route('/')
def index():
    return "Predict Fake Job Postings"
    
  
@application.route('/predict', methods=['GET'])
def predict():
    # best_model_nm = 'output/svm_text_squared_hinge_1e-05.pickle'
    best_model_nm = 'output/cnn_64_16_4.pickle'
    tokenizer_nm = 'output/tokenizer_64_16_4.pickle'
    # Load in best model
    best_model = pickle.load(open(best_model_nm, 'rb'))
    tokenizer = pickle.load(open(tokenizer_nm, 'rb'))

    json_request = request.get_json()
    if not json_request:
        return Response("No json provided.", status=400)
    text = json_request['text']
    if text is None:
        return Response("No text provided.", status=400)
    else:
        # preprocess text
        st = PorterStemmer()
        stopwords_dict = stopwords.words('english') # all stopwords in English
        text2 =  " ".join([st.stem(i) for i in text.split() if i not in stopwords_dict]).lower().translate(str.maketrans('', '', string.punctuation))

        # make prediction 
        encoded_text = tokenizer.texts_to_sequences([text2])
        in_text = pad_sequences(encoded_text, maxlen=1302, padding='post')
        pred_prob = best_model.predict(in_text)[0][0]
        corrected_prob = imbalance_correct(pred_prob=pred_prob) # imbalance correction

        result = {'Job Description':text, 'Probability': str(corrected_prob)}
        return flask.jsonify(result)

if __name__ == '__main__':
    application.run(debug=True, port = '5000', use_reloader=True)
