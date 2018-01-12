from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier

from util.feature_builder import *
from features.mean_wordembedding import *
from features.ngram import *
from features.contains import *
from transformers.data_extraction import *
from util.ngram import get_all_ngrams


def load_data(file_name, min_confidence=0.0, binary=False):
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    frame['raw_text'] = frame.apply(
        lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
        axis=1)
    if binary:
        frame['label'] = frame.apply(lambda row: row['label'] if row['label'] == 'NONE' else 'ARG', axis=1)
    # raw_text is wrong in the data, recreate it out of the one the annotators saw
    frame.to_csv('out1.csv')
    return shuffle(frame, random_state=42)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits)
    for train_index, test_index in k_fold.split(data['raw_text'],
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def build_feature_union(data):
    # all_mwe = ('all-mwe', make_pipeline(ExtractRawSentence(), MeanWordEmbedding()))
    # before_mwe = ('before-mwe', make_pipeline(ExtractFirstPart(), MeanWordEmbedding()))
    # middle_mwe = ('middle-mwe', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding()))
    # after_mew = ('after-mwe', make_pipeline(ExtractLastPart(), MeanWordEmbedding()))

    return FeatureUnion([
                            ('jjr-middle', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),  # 56 / 74
                            ('unigram-b', make_pipeline(ExtractFirstPart(), NGram(unigrams))),  # 62 / 81
                            ('unigram-e', make_pipeline(ExtractLastPart(), NGram(unigrams))),  # 62 / 81
                            ('bigram-middle', make_pipeline(ExtractMiddlePart(), NGram(bigrams, n=2, min_freq=2))),
                            # 66 / 68

                            #  ('trigram-middle', make_pipeline(ExtractMiddlePart(), NGram(trigrams, n=3))),  # 48 / 53

                            ('mwe-all', make_pipeline(ExtractRawSentence(), MeanWordEmbedding())),  # 60 / 77
                            # ('unigram-all', make_pipeline(ExtractRawSentence(), NGram(unigrams))),  # 59 / 79
                        ] + build_contains_word(
        ["better", "easier", "faster", "nicer", "wiser", "cooler", "decent", "safer", "superior", "solid", "teriffic",
         "worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior", "mediocre",
         "because", "more"
                    "?"],
        ExtractMiddlePart))


# 64 / 82

# ('bigram-middle-min-2', make_pipeline(ExtractMiddlePart(), NGram(bigrams, n=2, min_freq=2))),  # 40 / 40
#  ('unigram-middle-min-2', make_pipeline(ExtractMiddlePart(), NGram(unigrams, min_freq=2))),  # 41 / 42
#  ('jjr-whole', make_pipeline(ExtractRawSentence(), ContainsPos('JJR'))), # 40 / 64
# ('better-whole', make_pipeline(ExtractRawSentence(), ContainsWord('better'))), # 40 / 60
#  ('worse-whole', make_pipeline(ExtractRawSentence(), ContainsWord('worse'))), # 40 /45
# ('?-whole', make_pipeline(ExtractRawSentence(), ContainsWord('?'))), # 40 / 40

# ------ Classification
_labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
_labels_bin = ['ARG', 'NONE']

_data = load_data('out1.csv')
_data_bin = load_data('out1.csv', binary=True)

unigrams = get_all_ngrams(_data['raw_text'].values, n=1)
bigrams = get_all_ngrams(_data['raw_text'].values, n=2)
trigrams = get_all_ngrams(_data['raw_text'].values, n=3)


def run_pipeline(estimator, feature_union, data, labels):
    print(type(estimator))
    train, test = train_test_split(data, test_size=0.2, stratify=data['label'])
    print(len(train), len(test), len(data))
    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])

    fitted = pipeline.fit(train,
                          train['label'].values)

    predictions = fitted.predict(test)

    report = classification_report(test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(test['label'].values, predictions, labels=labels)
    print('RUN ------')
    print(report)
    print(conf)
    print(estimator.coef_)


def run_grid_search(estimator, feature_union):
    train, test = train_test_split(data, stratify=data['label'])
    print(len(train), len(test), len(data))

    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])
    print(pipeline.get_params().keys())

    scorer = make_scorer(f1_score, average='weighted')
    estimator.get_params().keys()
    param_grid = [
        {'features__all-1__ngram__n': [1], 'features__all-1__ngram__n_grams': [unigrams],
         'features__all-1__ngram__min_freq': [1, 2], 'features__b-1': [0, 1]},
        {'features__all-1__ngram__n': [2], 'features__all-1__ngram__n_grams': [bigrams],
         'features__all-1__ngram__min_freq': [1, 2]}]

    pipeline = GridSearchCV(pipeline, n_jobs=1, param_grid=param_grid, verbose=30, scoring=scorer, cv=2)

    fitted = pipeline.fit(train,
                          train['label'].values)
    predictions = fitted.predict(test)
    print("---------------------")
    print(pipeline.cv_results_)
    print("---------------------")
    print(pipeline.best_params_)
    print("---------------------")
    print(pipeline.best_score_)


# run_grid_search(LogisticRegression(), build_feature_union(data))

# run_pipeline(DummyClassifier(strategy='most_frequent', random_state=1337), build_feature_union(_data), _data, _labels)
run_pipeline(LogisticRegression(), build_feature_union(_data), _data, _labels)
print('######### BINARY #########')
# run_pipeline(DummyClassifier(strategy='most_frequent', random_state=1337), build_feature_union(_data_bin), _data_bin,
#             _labels_bin)
run_pipeline(LogisticRegression(n_jobs='4'), build_feature_union(_data), _data_bin, _labels_bin)
