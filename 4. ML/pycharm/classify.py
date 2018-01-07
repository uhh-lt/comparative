from features.mean_wordembedding import *
from features.contains_pos import *
from features.ngram import *
from transformers.data_extraction import *
from features.potato import *

from pandas import DataFrame as df

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.dummy import DummyClassifier

from bs4 import BeautifulSoup


def load_data(file_name, min_confidence=0.65):
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    # frame['raw_text'] = frame.apply(lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''), axis=1)
    # raw_text is wrong in the data, recreate it out of the one the annotators saw
    frame.to_csv('500_cleaned.csv')
    return shuffle(frame, random_state=42)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits)
    for train_index, test_index in k_fold.split(data['raw_text'],
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


# ('pre', SentenceSplit(data,'before_a')),
def build_feature_union(data):
    all_mwe = ('all-mwe', make_pipeline(ExtractRawSentence(), MeanWordEmbedding()))
    before_mwe = ('before-mwe', make_pipeline(ExtractFirstPart(), MeanWordEmbedding()))
    middle_mwe = ('middle-mwe', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding()))
    after_mew = ('after-mwe', make_pipeline(ExtractLastPart(), MeanWordEmbedding()))

    all_n = ('all-n', make_pipeline(ExtractRawSentence(), NGram(data['raw_text'].values)))
    before_n = ('before-n', make_pipeline(ExtractFirstPart(), NGram(data['raw_text'].values)))
    middle_n = ('middle-n', make_pipeline(ExtractMiddlePart(), NGram(data['raw_text'].values)))
    after_n = ('after-n', make_pipeline(ExtractLastPart(), NGram(data['raw_text'].values)))

    return FeatureUnion([
        all_mwe,
        before_mwe,
        middle_mwe,
        after_mew,

        all_n,
        before_n,
        middle_n,
        after_n
    ])


# ------ Classification
labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
data = load_data('500_cleaned.csv')  # [342:350]


def run_pipeline(estimator, feature_union):
    print(type(estimator))
    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])

    param_grid = {'foo' : [1]}
    pipeline = GridSearchCV(estimator=pipeline, param_grid=pipeline, verbose=30)


    fitted = pipeline.fit(train,
                          train['label'].values)
    predictions = fitted.predict(test)

    report = classification_report(
        test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(
        test['label'].values, predictions, labels=labels)
    return report, conf


for train, test in split_data(2, data):
    f = build_feature_union(data)
    # report_dummy, conf_dummy = run_pipeline(DummyClassifier(strategy='most_frequent'), f)
    # print('Baseline: Most Frequent Class\n', report_dummy)

    report, conf = run_pipeline(LogisticRegression(), f)
    print('Result\n', report, conf)
