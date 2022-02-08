import logging
import os
import warnings
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class Convert(mlflow.pyfunc.PythonModel):

    def delete_columns(self, df: pd.DataFrame):
        del df['SalesAmountInEuro']
        del df['time_delay_for_conversion']
        del df['product_price']
        return df

    # def replace_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
    #     return df.replace(-1, np.NaN).replace('-1', np.NaN)

    def click_to_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df['click_timestamp'] = df[['click_timestamp']].apply(
            lambda x: datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S").timestamp(), axis=1
        ).astype(int)
        return df

    def predict(self, context, model_input):
        df = model_input
        df = self.delete_columns(df)
        # df = self.replace_to_nan(df)
        df = self.click_to_timestamp(df)
        return df


if __name__ == "__main__":
    with mlflow.start_run() as run:
        model = Convert()
        model_path = os.path.join('pre_models', "convert-" + run.info.run_id)
        mlflow.pyfunc.save_model(path=model_path, python_model=model)
        loaded_model = mlflow.pyfunc.load_model(model_path)
        print('Run id is', run.info.run_id)
