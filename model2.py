import mlflow.pyfunc
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from random import randint
import logging
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow
from tensorflow.keras import optimizers
import keras.backend as K


def f1Score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
def nnmodel(X_train):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dense(64, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(64, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(64, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(64, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(128, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(128, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(128, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(128, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(128, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(64, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(64, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(64, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(10, Activation('relu')))
    model.add(BatchNormalization())
    model.add(Dense(1, Activation('sigmoid')))
    learning_rate = 0.001
    optimizer = optimizers.Adam(learning_rate)
    model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer,
                  metrics=[f1Score])
    return model
class Model(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = nnmodel()
        self.model.summary()


    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        history = self.model.fit(
            x,
            y,
            batch_size=256,
            epochs=100,
        )
    def predict(self, context, model_input:pd.DataFrame):
        if (model_input['FLAG']==0).sum():
            self.fit(
                model_input[model_input["FLAG"] == 0],
                model_input[model_input["FLAG"] == 0]['Sale'],
            )
        else:
            predicted = self.model.predict(model_input)
            predicted[predicted > 0.6] = 1
            predicted[predicted <= 0.6] = 0
            return predicted


with mlflow.start_run() as run:
    # Construct and save the model
    model = Model()
    model_path = os.path.join('models', "model2-" + run.info.run_id)
    mlflow.pyfunc.save_model(path=model_path, python_model=model)
    print('Model path is', model_path)
