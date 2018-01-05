from features.mean_wordembedding import *
from features.contains_pos import *
from features.ngram import *
from pandas import DataFrame as df

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score


def load_data(file_name):
    frame = df.from_csv(path='data/' + file_name)
    return shuffle(frame, random_state=42)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits)
    for train_index, test_index in k_fold.split(data['sentence'],
                                                data['assigned_class']):
        yield data.iloc[train_index], data.iloc[test_index]

def build_feature_union():
        mean_word_embedding = ('mean-wordembedding',
                                       Pipeline([('step-1',
                                                  MeanWordEmbedding())]))
        contains_jjr = ('contains-jjr', Pipeline([('step-1', ContainsPos('JJR'))]))
        unigram = ('unigram', Pipeline([('step-1', NGram(1,data['sentence'].values, min_freq=1))]))
        #bigram = ('bigram', Pipeline([('step-1', NGram(2,data['sentence'].values))]))
        #trigram = ('trigram', Pipeline([('step-1', NGram(3,data['sentence'].values))]))
        return FeatureUnion([
        #mean_word_embedding,
        #contains_jjr,
        unigram,
        #bigram,
        #trigram
        ])



# ------ Classification
labels = ['BETTER', 'WORSE', 'UNCLEAR',  'NO_COMP']
data = load_data('200-cw.csv')
#data = load_data('test.csv')

for train, test in split_data(2, data):
    feature_union = build_feature_union()
    pipeline = Pipeline([('features', feature_union), ('estimator', LogisticRegression())])


    fitted = pipeline.fit(train['sentence'].values, train['assigned_class'].values)
    predictions = fitted.predict(test['sentence'].values)

    report = classification_report(test['assigned_class'].values, predictions, labels=labels)
    conf = confusion_matrix(test['assigned_class'].values, predictions, labels=labels)
    print(report,'\n',conf,'\n------------------')
