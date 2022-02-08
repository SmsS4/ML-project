import logging
import os
import sys
import warnings

import mlflow
import pandas as pd

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class Download(mlflow.pyfunc.PythonModel):
    def __init__(self, url, val_split: float, test_split: float):
        self.val_split = val_split
        self.test_split = test_split
        self.url = url
        self.cache_path = "/var/tmp/downloaded_data.csv"

    def get_cache(self) -> pd.DataFrame:
        print('get cache')
        if os.path.exists(self.cache_path):
            return pd.read_csv(self.cache_path)
        return self.set_cache()

    def set_cache(self) -> pd.DataFrame:
        print('set cache')
        df = self.get_df()
        df.to_csv(self.cache_path)
        return df

    def add_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        v = int(self.val_split * len(df))
        t = int(self.test_split * len(df))
        df['FLAG'] = df.index.to_series().map(lambda x: 0 if x < len(df)-v-t else (1 if x < len(df)-t else 2))
        return df

    def get_df(self) -> pd.DataFrame:
        print('get df')
        url = 'https://drive.google.com/uc?id=' + self.url.split('/')[-2]
        return self.add_flag(pd.read_csv(url))[:100] #TODO

    def predict(self, context, model_input) -> pd.DataFrame:
        print('predict')
        return self.get_cache()


if __name__ == "__main__":
    with mlflow.start_run() as run:
        model = Download(
            'https://drive.google.com/file/d/1HBH9S64qfm_e2BNuCVDh_ipVaRmdufo7/view?usp=sharing',
            float(sys.argv[1]),
            float(sys.argv[2]),
        )
        model.set_cache()
        model_path = os.path.join('data_models',"download-" + run.info.run_id)
        mlflow.log_param('val_split', model.val_split)
        mlflow.log_param('test_split', model.test_split)
        mlflow.pyfunc.save_model(path=model_path, python_model=model)
        loaded_model = mlflow.pyfunc.load_model(model_path)
        print('Run id is', run.info.run_id)

