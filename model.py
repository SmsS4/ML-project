import mlflow.pyfunc
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


class Model(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input):
        return [0]


with mlflow.start_run() as run:
    # Construct and save the model
    model = Model()
    model_path = os.path.join('models', "model-"+run.info.run_id)
    mlflow.pyfunc.save_model(path=model_path, python_model=model)
    print('Model path is', model_path)
