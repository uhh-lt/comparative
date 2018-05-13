"""
This script tests all features on the held-out data.
"""

import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, make_pipeline
from xgboost import XGBClassifier

from features.contains_features import ContainsPos
from features.mean_embedding_feature import MeanWordEmbedding
from features.misc_features import SelectDataFrameColumn, POSTransformer
from transformers.data_extraction import ExtractMiddlePart
from util.precalculate_features import precalculate_embedding

MIDDLE_16 = 'middle_paths_unrestricted_16'
FULL_4 = 'full_paths_original_4'


def add_embeddings_to_df(df, embedding_file, column_name):
    embeddings = pd.read_csv(embedding_file)
    embeddings['embedding'] = embeddings.apply(lambda row: json.loads(row['embedding']), axis=1)
    df[column_name] = df.apply(lambda row: embeddings[embeddings.id == row['id']].embedding.values.tolist()[0], axis=1)
    return df


# adding the precalculated path embeddings and sentence embeddings
_train_data = pd.read_csv('data/data.csv')
train_data = add_embeddings_to_df(_train_data, 'data/path_embeddings/full_paths_original_4.csv', FULL_4)
train_data = add_embeddings_to_df(_train_data, 'data/path_embeddings/middle_paths_unrestricted_16.csv', MIDDLE_16)
train_data = precalculate_embedding(train_data)

_test_data = pd.read_csv('data/do-not-touch/held-out-data.csv')
test_data = add_embeddings_to_df(_test_data, 'data/do-not-touch/full_paths_original_4.csv', FULL_4)
test_data = add_embeddings_to_df(_test_data, 'data/do-not-touch/middle_paths_unrestricted_16.csv', MIDDLE_16)
test_data = precalculate_embedding(test_data)

print('Training Data Statistics: \n{}\n'.format(train_data.most_frequent_label.value_counts()))
print('Test Data Statistics: \n{}'.format(test_data.most_frequent_label.value_counts()))

feature_unions = [
    FeatureUnion([('Bag-Of-Words', make_pipeline(ExtractMiddlePart(), CountVectorizer()))]),
    FeatureUnion([('full_paths_original_4_aip', make_pipeline(SelectDataFrameColumn(FULL_4)))]),
    FeatureUnion([('middle_paths_unrestricted_16', make_pipeline(SelectDataFrameColumn(MIDDLE_16)))]),
    FeatureUnion([('InferSent', make_pipeline(SelectDataFrameColumn('embedding_middle_part', value_transform=lambda x: json.loads(x))))]),
    FeatureUnion([('Word Embedding', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding()))]),
    FeatureUnion([('POS n-grams', make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4))))]),
    FeatureUnion([('Contains JJR', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR')))]),
]

for union in feature_unions:
    classifier = XGBClassifier(n_jobs=8, n_estimators=1000)
    print(union)
    pipeline = make_pipeline(union, classifier)
    fitted = pipeline.fit(train_data, train_data['most_frequent_label'].values)
    predicted = fitted.predict(test_data)
    print(classification_report(test_data['most_frequent_label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
    print(confusion_matrix(test_data['most_frequent_label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE']))
