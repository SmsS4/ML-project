import logging
import os
import random
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class Transform(mlflow.pyfunc.PythonModel):
    def __init__(
            self,
            categorcals_to_numerics,
            nb_clicks_1week,
            click_timestamp,
            transform_product_pca_n: int,
    ):
        self.categorcals_to_numerics = categorcals_to_numerics
        self.nb_clicks_1week = nb_clicks_1week
        self.click_timestamp = click_timestamp
        self.transform_product_pca_n = transform_product_pca_n
        self.columns_pca = {}


    def split_col(self, data, col, add_to_df):
        """
        Args:
          col: to split
          add_to_df: True if you want to add splited to df
        """
        unq = data[col].unique()
        unq_df = pd.DataFrame()
        for u in tqdm(unq):
            unq_df[col + '_' + u] = (data[col] == u)
            if add_to_df:
                data[col + '_' + u] = (data[col] == u)
        if add_to_df:
            del data[col]
        return unq_df

    def fit_pca_transform(self, data, n, name):
        arr = np.reshape(data.to_numpy(), (len(data), len(data.columns)))
        pca = PCA(n_components=n)
        pca.fit(arr)
        self.columns_pca[name] = pca

    def pca_transform(self, data, append_to, name):
        arr = np.reshape(data.to_numpy(), (len(data), len(data.columns)))
        arr_pca = self.columns_pca[name].transform(arr)
        for i in range(arr_pca.shape[1]):
            append_to[f'{name}({i})'] = arr_pca[:, i]

    def fit(self, df: pd.DataFrame):
        print('fiting')
        for cat, pca_n in self.categorcals_to_numerics:
            if pca_n <= 0:
                continue
            splited = self.split_col(df, cat, False)
            self.fit_pca_transform(splited, pca_n, cat)
        self.fit_pca_transform(
            self.transform_product(df),
            self.transform_product_pca_n,
            'product',
        )
        print('fited')

    def transform_product(self, df):
        unq = pd.unique(df[[f'product_category({i + 1})' for i in range(7)]].values.ravel('K'))
        unq = unq[1:]
        new_df = pd.DataFrame()
        for u in tqdm(unq):
            new_df[u] = (
                    (df['product_category(1)'] == u) |
                    (df['product_category(2)'] == u) |
                    (df['product_category(3)'] == u) |
                    (df['product_category(4)'] == u) |
                    (df['product_category(5)'] == u) |
                    (df['product_category(6)'] == u) |
                    (df['product_category(7)'] == u)
            )
        return new_df

    def transform(self, df: pd.DataFrame):
        print('transforming')
        data = df.copy()
        for cat, pca_n in self.categorcals_to_numerics:
            print(cat, pca_n)
            if pca_n == 0:
                del data[cat]
                continue
            splited = self.split_col(data, cat, False)
            if pca_n != -1:
                self.pca_transform(splited, data, cat)
            elif pca_n == -1:
                for col in splited.columns:
                    data[col] = splited[col]
            del data[cat]
        self.pca_transform(
            self.transform_product(df),
            data,
            'product',
        )
        if self.click_timestamp:
            data['click_timestamp'] = df['click_timestamp']
        if self.nb_clicks_1week:
            data['nb_clicks_1week'] = df['nb_clicks_1week']
        for i in range(1, 8):
            del data[f'product_category({i})']
        return data

    def check_fit(self, df: pd.DataFrame):
        if (df['FLAG'] == 0).sum():
            print('train')
            self.fit(df.loc[(df['FLAG'] == 0)])

    def predict(self, context, model_input):
        df = model_input
        self.check_fit(df)
        df = self.transform(df)
        return df


if __name__ == "__main__":
    with mlflow.start_run() as run:
        mlflow.log_param('device_type', -1)
        mlflow.log_param('product_brand', 5)
        mlflow.log_param('product_country', -1)
        mlflow.log_param('product_title', 0)
        mlflow.log_param('partner_id', 5)
        mlflow.log_param('product', 10)
        mlflow.log_param('nb_clicks_1week', True)
        mlflow.log_param('click_timestamp', True)
        model = Transform(
            [
                ('device_type', -1),
                ('product_brand', 5),
                ('product_country', -1),
                ('product_title', 0),
                ('partner_id', 5),
            ],
            True,
            True,
            10,
        )
        model_path = os.path.join('pre_models', "transform" + run.info.run_id)
        mlflow.pyfunc.save_model(path=model_path, python_model=model)
        loaded_model = mlflow.pyfunc.load_model(model_path)
        print('Run id is', run.info.run_id)

