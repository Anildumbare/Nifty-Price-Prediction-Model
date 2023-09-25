import bentoml

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

data = pd.read_csv('FR 44 Model.csv')

data['Support Zone'] = data['Support Zone'].replace({'Y': 1, 'N': 0})
data['First Target'] = data['First Target'].replace({'Y': 1, 'N': 0})
data['Second target'] = data['Second target'].replace({'Y': 1, 'N': 0})
data['Third Target'] = data['Third Target'].replace({'Y': 1, 'N': 0})

data = data.dropna()
data = data.fillna(0)

data['Corrected Points'] = data['Corrected Points'].astype(int)
data['Support Zone'] = data['Support Zone'].astype(int)
data['First Target'] = data['First Target'].astype(int)
data['Second target'] = data['Second target'].astype(int)
data['Third Target'] = data['Third Target'].astype(int)
data['Total points Gain'] = data['Total points Gain'].astype(int)

label_encoder = LabelEncoder()
data['Opening'] = label_encoder.fit_transform(data['Opening'])
data['Correction Type'] = label_encoder.fit_transform(data['Correction Type'])
data['Overal Trend'] = label_encoder.fit_transform(data['Overal Trend'])
data['First Candel Colour'] = label_encoder.fit_transform(data['First Candel Colour'])

X = data[['Correction Type','Corrected Points','Overal Trend','Support Zone','First Target','Second target','Third Target']]
y = data['Total points Gain']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

base_models = [RandomForestRegressor(n_estimators=100, random_state=1),
               GradientBoostingRegressor(n_estimators=100, random_state=1)]

for model in base_models:
    model.fit(X_train, y_train)

base_model_predictions = [model.predict(X_test) for model in base_models]


meta_model = LinearRegression()
meta_model.fit(np.column_stack(base_model_predictions), y_test)

#test_base_model_predictions = [model.predict(X_test) for model in base_models]

#final_predictions = meta_model.predict(np.column_stack(test_base_model_predictions))

saved_model = bentoml.sklearn.save_model("fib_retest",meta_model)

print(f'Model Saved: {saved_model}')

