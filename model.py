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

from torch.nn.modules.activation import Sigmoid
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class DatasetCriteo(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = torch.tensor(self.X[idx,], dtype=torch.float32)

        return data, self.y[idx]


class MLP_Criteo(nn.Module):
    def __init__(self):
        super().__init__()
        listLayers = [
            nn.Linear(41, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()]
        for i in range(7):
            listLayers.append(nn.Linear(64, 64))
            listLayers.append(nn.BatchNorm1d(64))
            listLayers.append(nn.ReLU())
        listLayers.append(nn.Linear(64, 10))
        listLayers.append(nn.BatchNorm1d(10))
        listLayers.append(nn.ReLU())
        listLayers.append(nn.Linear(10, 1))
        listLayers.append(nn.Sigmoid())
        self.layers = nn.Sequential(
            *listLayers
        )
    def forward(self, x):
        x = self.layers(x)
        return x


class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, add_one: int = 10, batch_size: int = 256):
        self.add_one = add_one
        self.batch_size = batch_size
        self.model = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame):
        valid_dataset = DatasetCriteo(X=x_val.to_numpy(), y=y_val.to_numpy())
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

        x = x.to_numpy()
        y = y.to_numpy()
        x_copy = x.copy()
        y_copy = y.copy()
        for i in range(self.add_one):
            x_copy = np.append(x_copy, x[y == 1], 0)
            y_copy = np.append(y_copy, y[y == 1], 0)

        train_dataset = DatasetCriteo(X=x_copy, y=y_copy)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        model2 = MLP_Criteo()
        self.model = model2
        model2 = MLP_Criteo()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model2 = model2.to(device)
        optimizer = torch.optim.SGD(model2.parameters(), lr=0.1, momentum=0.8)
        loss_fn = nn.BCELoss()
        mean_train_losses = []
        mean_valid_losses = []
        mean_valid_fScores = []
        mean_train_fScores = []
        valid_acc_list = []
        epochs = 100

        for epoch in range(epochs):
            model2.train()

            train_losses = []
            valid_losses = []
            train_fScores = []
            valid_fScores = []
            for i, (datas, labels) in enumerate(train_loader):
                datas, labels = datas.to(device), labels.to(device)
                outputs = model2(datas)
                loss = loss_fn(outputs.to(torch.float32), labels.unsqueeze(1).to(torch.float32))
                predicted = torch.round(outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                fScore = f1_score(predicted.detach().numpy(), labels)
                train_fScores.append(fScore)

            model2.eval()

            with torch.no_grad():
                for i, (datas, labels) in enumerate(valid_loader):
                    datas, labels = datas.to(device), labels.to(device)
                    outputs = model2(datas)
                    loss = loss_fn(outputs.to(torch.float32), labels.unsqueeze(1).to(torch.float32))
                    valid_losses.append(loss.item())
                    predicted = torch.round(outputs)
                    valid_fScores.append(f1_score(predicted.detach().numpy(), labels))

            mean_train_losses.append(np.mean(train_losses))
            mean_valid_losses.append(np.mean(valid_losses))

            print(
                'epoch : {}, train loss : {:.4f}, train f1score : {:.4f}' \
                    .format(epoch + 1, np.mean(train_losses), np.mean(train_fScores)))
    def predict(self, context, model_input:pd.DataFrame):
        if (model_input['FLAG']==0).sum():
            self.fit(
                model_input[model_input["FLAG"] == 0],
                model_input[model_input["FLAG"] == 0]['Sale'],
                model_input[model_input["FLAG"] == 1],
                model_input[model_input["FLAG"] == 1]['Sale'],
            )
        else:
            self.model.eval()
            X_test_tensor = torch.tensor(model_input, dtype=torch.float32)
            with torch.no_grad():
                return self.model(X_test_tensor)


with mlflow.start_run() as run:
    # Construct and save the model
    model = Model()
    model_path = os.path.join('models', "model-" + run.info.run_id)
    mlflow.pyfunc.save_model(path=model_path, python_model=model)
    print('Model path is', model_path)
