from features.mean_wordembedding import *
from features.contains_pos import *
from features.ngram import *

from transformers.sentence_split import *

from pandas import DataFrame as df

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.dummy import DummyClassifier


data = frame = df.from_csv(path='data/brands_500.csv')

feature_union = FeatureUnion([('feat1', make_pipeline(MeanWordEmbedding()))])
pipeline = Pipeline([('features', feature_union),  ('estimator',
                                                   LogisticRegression)])
fitted = pipeline.fit(data['raw_text'].values, data['label'].values)
predictions = fitted.predict(data['raw_text'].values)
