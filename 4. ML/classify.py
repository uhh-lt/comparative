from features.mean_wordembedding import *
from features.contains_pos import *
from pandas import DataFrame as df

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def load_data(file_name):
    frame = df.read_csv(path='data/' + file_name)
    return shuffle(frame, random_state=42)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits)
    for train_index, test_index in k_fold.split(data['sentence'],
                                                data['assigned_class']):
        yield data.iloc[train_index], data.iloc[test_index]


# ------ Classification
labels = ['BETTER', 'WORSE', 'UNCLEAR',  'NO_COMP']
data = load_data('200-cw.csv')

for train, test in split_data(2, data):
    feature_union = FeatureUnion([('mean-wordembedding',
                                   Pipeline([('step-1',
                                              MeanWordEmbedding())]))])

    pipeline = Pipeline([('features', feature_union), ('estimator', LogisticRegression())])
    fitted = pipeline.fit(train['sentence'].values, train['assigned_class'].values)
    predictions = fitted.predict(test['sentence'].values)

    report = classification_report(test['assigned_class'].values, predictions, labels=labels)
    conf = confusion_matrix(test['assigned_class'].values, predictions, labels=labels)
    print(report,'\n',conf,'\n------------------')
