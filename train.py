import pandas as pd
from autogluon.tabular import TabularPredictor

# Caminho relativo para o arquivo CSV
path_crawleds = 'databases/crawleds.csv'    
path_merged = 'databases/mergeds.csv'

train_data = pd.read_csv(path_merged)
train_data_efficiency = train_data.drop(['complexity_class'], axis=1)
train_data_class = train_data

predictor_efficiency  = TabularPredictor(label='efficiency').fit(train_data_efficiency, presets=['high_quality'])
predictor_class = TabularPredictor(label='complexity_class').fit(train_data_class, presets=['high_quality'])
