"""
This script tests the path embeddings generated with the original code (max. length of four;
the first object must be reachable from the lowest common head by following left edges only; right object by right edges only).
This setup did not return paths for every sentence, so all sentences without a path are excluded here.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from infersent.infersent_feature import initialize_infersent, InfersentFeature
from util.data_utils import k_folds

data_filtered = pd.read_csv('data/data_if.csv')
path_embeddings = pd.read_csv('data/path_embeddings/full_paths_original_4.csv')
print('All sentences {}'.format(len(path_embeddings)))
print('Without embedding {}'.format(len(path_embeddings[path_embeddings.paths.str.contains('NOPATH')])))
path_embeddings = path_embeddings[path_embeddings.paths.str.contains('NOPATH') == False]
print('With sentences {}'.format(len(path_embeddings)))
data_filtered = data_filtered[data_filtered.id.isin(path_embeddings.id.values.tolist())]
data_filtered['feat'] = data_filtered.apply(lambda x: json.loads(path_embeddings[path_embeddings.id == x['id']].embedding.values.tolist()[0]), axis=1)

print(data_filtered.most_frequent_label.value_counts())



all_predictions = []
all_test_vals = []
for train, test in k_folds(2, data_filtered, random_state=42):
    classifier = XGBClassifier(n_jobs=8, n_estimators=1000)
    array = np.array(train['feat'].values.tolist())
    fitted = classifier.fit(array, train['most_frequent_label'].values)
    predict = fitted.predict(np.array(test['feat'].values.tolist()))
    all_predictions += predict.tolist()
    all_test_vals += test['most_frequent_label'].values.tolist()

data = pd.read_csv('data/data_if.csv')
data_filtered_2 = data[data.id.isin(data_filtered.id.values.tolist()) == False]
infersent_model = initialize_infersent(data_filtered_2.sentence.values)
assert (len(data_filtered) + len(data_filtered_2)) == len(data)

for train, test in k_folds(2, data_filtered_2, random_state=42):
    classifier = XGBClassifier(n_jobs=8, n_estimators=1000)
    f = InfersentFeature(infersent_model)
    train_f = f.transform(train.sentence.values)
    test_f = f.transform(test.sentence.values)
    values = train['most_frequent_label'].values.tolist()
    fitted = classifier.fit(train_f, values)
    all_predictions += fitted.predict(test_f).tolist()
    all_test_vals += test['most_frequent_label'].values.tolist()

print(
    classification_report(all_test_vals, all_predictions, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
