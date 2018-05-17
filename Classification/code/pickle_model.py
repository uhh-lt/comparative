import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from infersent.infersent_feature import InfersentFeature
from transformers.data_extraction import ExtractMiddlePart

train = pd.read_csv('data/data.csv')[:50]
test = pd.read_csv('data/do-not-touch/held-out-data.csv')

pl = make_pipeline(ExtractMiddlePart(), InfersentFeature(), XGBClassifier(n_jobs=8, n_estimators=1000))
fitted = pl.fit(train, train['most_frequent_label'].values)
predicted = fitted.predict(test)

print(classification_report(test['most_frequent_label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))

saved = joblib.dump(fitted, 'sentence_embeddings.gz', compress=3)

loaded = joblib.load('sentence_embeddings.pkl')

test_df = pd.DataFrame(columns=['object_a', 'object_b', 'sentence'])
test_df.loc[0] = ['Python', 'Ruby', 'Python is better than Ruby']
print(loaded.predict(test_df))
