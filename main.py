import json
import os
from pprint import pprint
from typing import Optional

import mlflow
import pandas as pd

import requests


class Routes:
    DOWNLOAD = "http://127.0.0.1:1234"
    CONVERT = "http://127.0.0.1:1235"
    CLEAN = "http://127.0.0.1:1236"
    PCA = "http://127.0.0.1:1237"
    MODEL = "http://127.0.0.1:1238"


def df_to_json(df: pd.DataFrame) -> dict:
    if df.isna().sum().sum():
        print('>>>Warning!!! data has nan<<<')
    return {
        "columns": list(df.columns),
        "data": df.fillna('-1').values.tolist(),
    }


def json_to_df(res: requests.Response) -> pd.DataFrame:
    if res.status_code != 200:
        print(res.status_code, res.content)
        pprint(json.loads(res.content.decode("utf-8")))
        exit(1)
    df = pd.DataFrame(res.json())
    if 'Unnamed: 0' in df:
        del df['Unnamed: 0']
    return df


class Stage:
    TOTAL = 0
    CALLED = 0

    def __init__(
            self,
            name: str,
            route: str,
    ):
        Stage.TOTAL += 1
        print('Stage', name, 'crated')
        self.route = route
        self.name = name
        self.next: Optional['Stage'] = None

    def set_next(self, stage: 'Stage') -> 'Stage':
        self.next = stage
        return self

    def __call__(self, data: pd.DataFrame):
        Stage.CALLED += 1
        print(f'Stage {self.name} called {Stage.CALLED}/{Stage.TOTAL}')
        print(f"{self.route}/invocations")
        result = requests.post(
            f"{self.route}/invocations",
            headers={
                'Content-Type': 'application/json'
            },
            json=df_to_json(data),
        )
        df = json_to_df(result)
        if self.next:
            return self.next(df)
        return df


pre = Stage('convert', Routes.CONVERT).set_next(
    Stage('clean_data', Routes.CLEAN).set_next(
        Stage('pca', Routes.PCA)
    )
)
download_and_pre = Stage('download', Routes.DOWNLOAD).set_next(pre)
model = Stage('model', Routes.MODEL)
train_pipeline = lambda: model(download_and_pre(pd.DataFrame()))

test_pipeline = lambda data: model(download_and_pre(data))


class CallRealModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return test_pipeline(model_input)


def main():
    with mlflow.start_run() as run:
        print('Training...')
        caller_model = CallRealModel()
        train_pipeline()
        model_path = os.path.join('models', "pipeline-" + run.info.run_id)
        mlflow.pyfunc.save_model(path=model_path, python_model=caller_model)
        print('Pipeline saved in', model_path)


if __name__ == "__main__":

    main()
