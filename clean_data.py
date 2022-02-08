import logging
import os
import random
import warnings

import mlflow
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class CleanData(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.means = {}
        self.fill_objects = {}

    def delete_too_many(self, df: pd.DataFrame) -> pd.DataFrame:
        # diffrent values
        del df['user_id']
        del df['product_id']
        # nan
        del df['product_age_group']
        del df['product_gender']
        del df['audience_id']
        return df

    def fit(self, df: pd.DataFrame):
        print('Fiting clean data')
        for column in df.iloc[:, list(df.dtypes != 'object')]:
            self.means[column] = df[column].mean()

        df_filles = df.copy()
        for column in df_filles.loc[:, list(df_filles.dtypes == 'object')]:
            self.fill_objects[column] = list(df_filles.loc[df_filles[column].notna()][column])

    def transform(self, df: pd.DataFrame):
        df = df.replace(-1, np.NaN).replace('-1', np.NaN)
        for column in df.loc[:, list(df.dtypes != 'object')]:
            df[column][df[column].isnull()] = self.means[column]
        for column in df.loc[:, list(df.dtypes == 'object')]:
            if not df[column].isna().sum():
                continue
            x = df[column]
            x[df[column].isna()] = random.choices(
                self.fill_objects[column],
                k=(df[column].isna().sum())
            )
            df[column] = x
        # outliers
        df = df.drop(df[df['nb_clicks_1week'] > 10000].index)
        return df

    def check_fit(self, df: pd.DataFrame):
        if (df['FLAG'] == 0).sum():
            self.fit(df.loc[(df['FLAG'] == 0)])

    def predict(self, context, model_input):
        df = model_input
        df = self.delete_too_many(df)
        self.check_fit(df)
        df = self.transform(df)
        return df


if __name__ == "__main__":
    with mlflow.start_run() as run:
        model = CleanData()
        model_path = os.path.join('pre_models', "clean-" + run.info.run_id)
        mlflow.pyfunc.save_model(path=model_path, python_model=model)
        loaded_model = mlflow.pyfunc.load_model(model_path)
        print('Run id is', run.info.run_id)
