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


class AddN(mlflow.pyfunc.PythonModel):
    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)


with mlflow.start_run() as run:
    # Construct and save the model
    model_path = "add_n_model"
    add5_model = AddN(n=5)
    model_path = os.path.join('models', run.info.run_id)
    mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)

    print(os.path.join(os.getcwd(), "models", run.info.run_id))

    # Load the model in `python_function` format
    loaded_model = mlflow.pyfunc.load_model(model_path)

    # Evaluate the model
    model_input = pd.DataFrame([range(10)])
    model_output = loaded_model.predict(model_input)
    assert model_output.equals(pd.DataFrame([range(5, 15)]))
    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # # Model registry does not work with file store
    # if tracking_url_type_store != "file":
    #     # Register the model
    #     # There are other ways to use the Model Registry, which depends on the use case,
    #     # please refer to the doc for more information:
    #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #     mlflow.pyfunc.log_model(add5_model, "model", registered_model_name="Pre")
    # else:
    #     mlflow.pyfunc.log_model(add5_model, "model")
