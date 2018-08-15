import torch

import pandas as pd
from imp import reload
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline, FeatureUnion

from features.misc_features import SelectDataFrameColumn
from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractMiddlePart
from util.data_utils import load_data
from util.precalculate_features import precalculate_embedding

data = pd.concat([load_data('../all_data_files/experiments/data.csv'), load_data('../all_data_files/held-out/held-out-data.csv')])

print(data.domain.value_counts())

train_on = ['jbt', 'compsci', 'brands']
LABEL = 'most_frequent_label'

data = precalculate_embedding(data)

for domain in train_on:
    train = data[data.domain == domain]
    test = data[data.domain != domain]
    print("Train Set Size {}".format(len(train)))
    print("Test Set Size {}".format(len(test)))

    print("Train on: {} | Test on {}".format(domain, [i for i in train_on if i != domain]))

    classifier = LogisticRegression()
    pipeline = make_pipeline(SelectDataFrameColumn('embedding_middle_part'), classifier)
    fitted = pipeline.fit(train, train[LABEL].values)
    predicted = fitted.predict(test)
    print(classification_report(test[LABEL].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
    print("==================================")
