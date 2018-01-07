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
        #
        all_n,
        before_n,
        middle_n,
        after_n
    ])


# ------ Classification
labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
data = load_data('500_cleaned.csv')  # [342:350]


def run_pipeline(estimator, feature_union):
    train, test = train_test_split(data, stratify=data['label'])

    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])
    scorer = make_scorer(f1_score, average='weighted')

    fitted = pipeline.fit(train,
                          train['label'].values)

    predictions = fitted.predict(test)

    report = classification_report(test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(test['label'].values, predictions, labels=labels)
    print('RUN ------')
    print(report)
    print(conf)


def run_grid_search(estimator, feature_union):
    train, test = train_test_split(data, stratify=data['label'])
    print(len(train), len(test), len(data))

    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])
    print(pipeline.get_params().keys())

    scorer = make_scorer(f1_score, average='weighted')

    param_grid = {
        'features__before-n__ngram__n': [1, 2, 3],
        'features__middle-n__ngram__n': [1, 2, 3],
        'features__after-n__ngram__n': [1, 2, 3],
        'features__all-n__ngram__n': [1, 2, 3],

        'features__before-n__ngram__min_freq': [1, 2],
        'features__middle-n__ngram__min_freq': [1, 2],
        'features__after-n__ngram__min_freq': [1, 2],
        'features__all-n__ngram__min_freq': [1, 2],
        'features__transformer_weights': [{'all-n': 0}, {'all-n': 1}]
        #   'features__transformer_weights': [{'all-mwe': 0}, {'all-mwe': 1}, {'before-mwe': 0}, {'before-mwe': 1},
        #                                    {'middle-mwe': 0}, {'middle-mwe': 1}, {'after-mwe': 0}, {'after-mwe': 1},
        #                                   {'all-mwe': 0, 'before-mwe': 0, 'after-mwe': 0, 'middle-mwe': 0}]
    }

    pipeline = GridSearchCV(pipeline, param_grid=param_grid, verbose=30, scoring=scorer, cv=2)

    fitted = pipeline.fit(train,
                          train['label'].values)
    predictions = fitted.predict(test)
    print("---------------------")
    print(pipeline.cv_results_)
    print("---------------------")
    print(pipeline.best_params_)
    print("---------------------")
    print(pipeline.best_score_)


run_grid_search(LogisticRegression(), build_feature_union(data))
run_pipeline(LogisticRegression(), build_feature_union(data))
