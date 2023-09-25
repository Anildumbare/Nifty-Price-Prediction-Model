import bentoml

model_runner = bentoml.sklearn.get('fib_retest').to_runner()
model_runner.init_local()
print(model_runner.predict.run([[75,1]]))

# OR TRY THIS
"""import bentoml
import pandas as pd

@bentoml.artifacts([bentoml.sklearn.Model("model")])
class MyScikitLearnService(bentoml.BentoService):

    @bentoml.api(input_type=pd.DataFrame, batch=False)
    def predict(self, df):
        return self.artifacts.model.predict(df)"""

"""from bentoml import load

loaded_service = load("/path/to/your/saved/bentoml/service")"""

"""test_data = pd.DataFrame(...)  # Replace with your test data"""

"""predictions = loaded_service.predict(test_data)"""

