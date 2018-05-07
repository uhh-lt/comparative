"""
This script tests the path embeddings generated with the original code (max. length of four;
the first object must be reachable from the lowest common head by following left edges only; right object by right edges only).
This setup did not return paths for every sentence, so all sentences without a path are excluded here.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from util.data_utils import k_folds

data = pd.read_csv('data/data_if.csv')
path_embeddings = pd.read_csv('data/path_embeddings/full_paths_original_4.csv')
print('All sentences {}'.format(len(path_embeddings)))
print('Without embedding {}'.format(len(path_embeddings[path_embeddings.paths.str.contains('NOPATH')])))
path_embeddings = path_embeddings[path_embeddings.paths.str.contains('NOPATH') == False]
print('With sentences {}'.format(len(path_embeddings)))
data = data[data.id.isin(path_embeddings.id.values.tolist())]
data['feat'] = data.apply(lambda x: json.loads( path_embeddings[path_embeddings.id == x['id']].embedding.values.tolist()[0]), axis=1)

print(data.most_frequent_label.value_counts())

classifier = XGBClassifier(n_jobs=8, n_estimators=1000)

for train, test in k_folds(2, data, random_state=42):
    array = np.array(train['feat'].values.tolist())
    fitted = classifier.fit(array, train['most_frequent_label'].values)
    predicted = fitted.predict(np.array(test['feat'].values.tolist()))
    print(
        classification_report(test['most_frequent_label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
