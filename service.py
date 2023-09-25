import numpy as np
import bentoml
import pandas as pd
from bentoml.io import NumpyNdarray

model_runner = bentoml.sklearn.get('fib_retest').to_runner()

svc = bentoml.Service("fr_clasifire",runners=[model_runner])

@svc.api(input=NumpyNdarray(shape=(2,2), enforce_shape=False), output=NumpyNdarray())
def predict(input_array: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_array)
    return result
