from typing import Optional


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

    def call(self, data: pd.DataFrame):
        Stage.CALLED += 1
        print(f'Stage {self.name} called {Stage.CALLED}/{Stage.TOTAL}')
        result = requests.post(
            f"{self.route}/invocations",
            headers={
                'Content-Type': 'application/json'
            },
            json=df_to_json(data),
        )
        df = json_to_df(result)
        if self.next:
            return self.next.call(df)
        return df