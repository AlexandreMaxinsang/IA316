from flask import Flask, request, jsonify
import tensorflow as tf
import requests
import pandas as pd
from Recommeder import Recommender_embedding

app = Flask(__name__)
model = None


user_id = 'IAHIZPIW80WPGDW7P7JE'
base_url = "http://52.47.62.31"
url_reset = base_url + "/reset"
url_predict = base_url + "/predict"
params = {"user_id" : user_id}

def getdata(url_reset,url_predict,params) :

    r = requests.get(url=url_reset, params=params)
    data = r.json()
    user_history = data["user_history"]
    item_history = data["item_history"]
    rating_history = data["rating_history"]
    next_user = data["next_user"]
    next_item = data["next_item"]
    train_data = pd.DataFrame({'user_id': user_history, 'item_id': item_history, 'rating': rating_history})

    return train_data,next_user,next_item


@app.route("/")
def home():
    return "First Environement"

@app.route("/train")
def train():
    global model
    train_data,next_user,next_item = getdata(url_reset,url_predict,params)
    train_data,next_user,next_item = getdata(url_reset,url_predict,params)
    model = Recommender_embedding(40)
    model.fit_embedding(train_data[['user_id','item_id']],train_data[['rating']])

    #Save the graph of the model
    global graph
    graph = tf.get_default_graph()
    return "Trained"

@app.route("/predict", methods=['GET','POST'])
def predict():

    input1 = request.args.get('input1')
    input2 = request.args.get('input2')

    with graph.as_default():
        predict = model.predict_embedding([[input1],[input2]])
    d = {"predict" : str(predict)}
    return jsonify(d)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
