import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from trustee import ClassificationTrustee as ct 
from trustee.report.trust import TrustReport as tr
from sklearn.metrics import classification_report

# Caminho relativo para o arquivo CSV
path_crawleds = 'databases/crawleds.csv'    
path_merged = 'databases/mergeds.csv'
path_merged_encoded = 'databases/mergeds_encoded.csv'

train_data = pd.read_csv(path_merged)
train_data_efficiency = train_data.drop(['complexity_class'], axis=1)

train_data_class = train_data

#predictor_efficiency = TabularPredictor.load("AutogluonModels/efficiency_classifier")
#predictor_efficiency = predictor_efficiency._get_model_best()
#predictor_class = TabularPredictor.load("AutogluonModels/complexity_class_classifier")
#predictor_class = predictor_class._get_model_best()

path_efficiency_trustee = "AutogluonModels/efficiency_classifier_trustee/models/"
path_complexity_class_trustee = "AutogluonModels/complexity_class_classifier_trustee/models/"

trustee_data = pd.read_csv(path_merged_encoded)
train_data_efficiency_trustee = trustee_data.drop(['complexity_class', 'filename'], axis=1)
train_data_class_trustee = trustee_data.drop(['filename'], axis=1)

predictor_efficiency_trustee = TabularPredictor(label='efficiency', path = path_efficiency_trustee).fit(train_data_efficiency_trustee, presets=['medium_quality'])
# Obter o melhor modelo do predictor_efficiency_trustee
best_model_efficiency_trustee = predictor_efficiency_trustee.set_model_best("LightGBM")
y_pred_efficiency = predictor_efficiency_trustee.predict(train_data_efficiency_trustee)
predictor_class_trustee = TabularPredictor(label='complexity_class', path=path_complexity_class_trustee).fit(train_data_class_trustee, presets=['medium_quality'])
best_model_complexity_class_trustee = predictor_class_trustee.set_model_best("LightGBM")
y_pred_class = predictor_class_trustee.predict(train_data_class_trustee)



trustee_efficiency = ct(expert= predictor_efficiency_trustee) #trustee efficiency class
trustee_efficiency.fit(train_data_efficiency_trustee, train_data_efficiency_trustee['efficiency'], samples_size = 0.2, verbose = True) #trustee efficiency class
decision_tree, pruned_decision_tree, agreement, reward = trustee_efficiency.explain()
decision_tree_y_pred_efficiency = decision_tree.predict(train_data_efficiency_trustee)

print("Model explanation global fidelity report:")
print(classification_report(y_pred_efficiency, decision_tree_y_pred_efficiency))
#print("Model explanation score report:")
#print(classification_report(trustee_data_efficiency['efficiency'], decision_tree_y_pred))

trustee_class = ct(expert= predictor_class_trustee) #trustee complexity class
trustee_class.fit(train_data_class_trustee, train_data_class_trustee['complexity_class'], samples_size = 0.2, verbose = True) #trustee class class
decision_tree, pruned_decision_tree, agreement, reward = trustee_class.explain()
decision_tree_y_pred_class = decision_tree.predict(train_data_class_trustee)

print("Model explanation global fidelity report:")
print(classification_report(y_pred_class, decision_tree_y_pred_class))
#print("Model explanation score report:")
#print(classification_report(trustee_data_efficiency['efficiency'], decision_tree_y_pred))

trust_report = tr(predictor_efficiency_trustee, X = train_data_efficiency_trustee, y= train_data_efficiency_trustee['efficiency'])
print(trust_report)
trust_report.generate_report()