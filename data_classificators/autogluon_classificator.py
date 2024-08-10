import json
import os
import warnings

import pandas as pd
from autogluon.tabular import TabularPredictor


from databases import *

warnings.filterwarnings("ignore")

path_merged = 'databases/mergeds.csv'

train_data = pd.read_csv(path_merged)
train_data_efficiency = train_data.drop(['complexity_class'], axis=1)
train_data_class = train_data



def autogluon_classifier(code):
    predictor_class = TabularPredictor.load("AutogluonModels/ag-20240324_213528") #autogluon classificator class
    
    predictor_efficiency = TabularPredictor.load("AutogluonModels/ag-20240324_213451") #autogluon classificator efficiency
    # dt, pruned_dt, agreement, reward = trustee.explain()
    # dt_y_pred = dt.predict(X_test)

    # print("Model explanation global fidelity report:")
    # print(classification_report(y_pred, dt_y_pred))
    # print("Model explanation score report:")
    # print(classification_report(y_test, dt_y_pred))
    

    predictions_efficiency = predictor_efficiency.predict(code) #autogluon predict efficiency
    
    code['efficiency'] = predictions_efficiency[0]  #add autogluon predicted efficiency to code

    predictions_class = predictor_class.predict(code) #autogluon predict class
    
    code['complexity_class'] = predictions_class[0] #add autogluon predicted class to code

    
    csv_file = str(code['filename'])
    csv_file = csv_file.replace('.java', '.csv')
    # test_data.to_csv(csv_file, index=False)
    
    # Nome da pasta que você deseja criar
    folder_name = 'csv_files'

    # Cria a pasta se ela não existir
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # Cria o caminho completo para o arquivo CSV dentro da pasta
    #csv_file_path = os.path.join(folder_name, csv_file)

    # Salva o DataFrame em um arquivo CSV dentro da pasta
    #code.to_csv(csv_file_path, index=False)
    
    efficiency_prediction = predictions_efficiency[0]   #autogluon predicted efficiency
    class_prediction = predictions_class[0] #autogluon predicted class
    result = {'Efficiency': efficiency_prediction, 'Complexity class': class_prediction}
    result_json = json.dumps(result)
    return result_json



