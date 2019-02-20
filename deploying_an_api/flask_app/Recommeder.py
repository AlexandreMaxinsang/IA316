import requests,time
import numpy as np
import pandas as pd

from keras.layers import Input, Embedding, Flatten, Concatenate, Dropout, Dense
from keras.models import Model

class Recommender_embedding :
    """


    """
    def __init__(self, embedding_size,batch_size=64,epochs=50 ,shuffle =True):
        self.model = None
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

    def fit_embedding(self, train_x, train_y):

        self.nb_users = train_x['user_id'].nunique()
        self.nb_items = train_x['item_id'].nunique()

        # Model
        user_id_input = Input(shape=[1],name='user')
        item_id_input = Input(shape=[1], name='item')
        user_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.nb_users + 1,
                               input_length=1, name='user_embedding')(user_id_input)
        item_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.nb_items + 1,
                               input_length=1, name='item_embedding')(item_id_input)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)
        input_vecs = Concatenate()([user_vecs, item_vecs])
        x = Dense(64, activation='relu')(input_vecs)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        y = Dense(1)(x)

        self.model = Model(inputs=[user_id_input, item_id_input], outputs=y)
        self.model.compile(optimizer='adam', loss='MSE')
        self.model.fit([train_x["user_id"], train_x["item_id"]], train_y,
            steps_per_epoch = 1000, epochs=self.epochs,validation_split =0.1,validation_steps = 1000 ,shuffle=self.shuffle)

    def predict_embedding(self, test_x):
        return self.model.predict(test_x)

    def evaluation(self,url_predict,next_user,next_item,params,nb_samples = 50):

        mse, mae = 0, 0
        for i in range(nb_samples) :
            time.sleep(0.5)
            test_x = [[next_user], [next_item]]
            params["predicted_score"] = self.model.predict_embedding(test_x)
            prediction = params["predicted_score"]
            r = requests.get(url=url_predict, params=params)
            d = r.json()
            rating = d["rating"]
            next_user = d["next_user"]
            next_item = d["next_item"]
            mse += (rating - prediction) ** 2
            mae += abs(rating - prediction)

        return mse/nb_samples,mae/nb_samples


if __name__ == '__main__':

    user_id = 'Y6EKWA0GK1D0VCTN0RT7'
    base_url = "http://52.47.62.31"
    url_reset = base_url + "/reset"
    url_predict = base_url + "/predict"
    params = {"user_id" : user_id}

    r = requests.get(url=url_reset, params=params)
    data = r.json()
    user_history = data["user_history"]
    item_history = data["item_history"]
    rating_history = data["rating_history"]
    next_user = data["next_user"]
    next_item = data["next_item"]
    train_data = pd.DataFrame({'user_id': user_history, 'item_id': item_history, 'rating': rating_history})


    model = Recommender_embedding(40)
    model.fit_embedding(train_data[['user_id','item_id']],train_data[['rating']])

    print(model.evaluation(url_predict,next_user,next_item, params))
