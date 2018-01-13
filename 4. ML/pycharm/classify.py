from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import threading
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from features.counts import PunctuationCount
from util.feature_builder import *
from features.mean_wordembedding import *
from features.ngram import *
from features.counts import *
from features.contains import *
from transformers.data_extraction import *
from util.ngram import get_all_ngrams
from itertools import combinations

CUE_WORDS_WORSE = ["worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior",
                   "mediocre"]

CUE_WORDS_BETTER = ["better", "easier", "faster", "nicer", "wiser", "cooler", "decent", "safer", "superior", "solid",
                    "teriffic"]


def load_data(file_name, min_confidence=0.6, binary=False):
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    frame['raw_text'] = frame.apply(
        lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
        axis=1)
    if binary:
        frame['label'] = frame.apply(lambda row: row['label'] if row['label'] == 'NONE' else 'ARG', axis=1)
    # raw_text is wrong in the data, recreate it out of the one the annotators saw

    return shuffle(frame, random_state=42)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits)
    for train_index, test_index in k_fold.split(data['raw_text'],
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def build_feature_union(data):
    return FeatureUnion([
        ('unigram', Pipeline([('extractor', ExtractMiddlePart()), ('ngram', NGram(unigrams))])),
        ('punct', Pipeline([('extractor', ExtractMiddlePart()), ('counter', PunctuationCount())])),
        ('ent', Pipeline([('extractor', ExtractRawSentence()), ('counter', NECount())])),
        ('noun', Pipeline([('extractor', ExtractRawSentence()), ('counter', NounChunkCount())])),
        ('cue-word-a', make_pipeline(ExtractMiddlePart(), ContainsWord('foo'))),
        ('cue-word-b', make_pipeline(ExtractMiddlePart(), ContainsWord('foo')))

        # ('mwe-m', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding())),  # 68 65 66 67 57
        # ('unigram-', make_pipeline(ExtractRawSentence(), NGram(unigrams, count=True))),  # 63 62 45 62 58
        #  ('unigram', make_pipeline(ExtractRawSentence(), NGram(unigrams))),  # 63 62 45 60 55
        # ('bigram-m', make_pipeline(ExtractMiddlePart(), NGram(bigrams, n=2))),  # 62 67 45 69 62
        #  ('jjr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),
        # 61 61 61 61 61 -> alles entweder better or none
        # vec
        ##  ('vec', make_pipeline(ExtractRawSentence(), TfidfVectorizer())) # 61 63 45 63 53

    ]  # + build_contains_word(CUE_WORDS_WORSE, ExtractRawSentence)
    )


# ------ Classification


_labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
_labels_bin = ['ARG', 'NONE']

_data = load_data('train-data.csv')
_data_bin = load_data('train-data.csv', binary=True)

unigrams = get_all_ngrams(_data['raw_text'].values, n=1)
bigrams = get_all_ngrams(_data['raw_text'].values, n=2)


# trigrams = get_all_ngrams(_data['raw_text'].values, n=3)


def run_pipeline(estimator, feature_union, data, labels):
    print('******** ' + str(type(estimator)) + ' ********')
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])

    fitted = pipeline.fit(train,
                          train['label'].values)

    predictions = fitted.predict(test)

    report = classification_report(test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(test['label'].values, predictions, labels=labels)

    print(report)
    print(conf)


def run_grid_search(estimator, feature_union, data):
    train, test = train_test_split(data, stratify=data['label'])
    print(estimator.get_params())
    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])
   # print(pipeline.get_params().keys())

    scorer = make_scorer(f1_score, average='weighted')
    estimator.get_params().keys()
    param_grid = {
        'features__unigram__extractor': [ExtractRawSentence(), ExtractMiddlePart()],
        'features__ent__extractor': [ExtractRawSentence(), ExtractFirstPart(), ExtractLastPart()],
        'features__noun__extractor': [ExtractRawSentence(), ExtractFirstPart(), ExtractLastPart()],
        'features__punct__extractor': [ExtractRawSentence(), ExtractFirstPart(), ExtractLastPart()],
        'features__cue-word-a__containsword__word': list(combinations(CUE_WORDS_BETTER, 3)),
        'features__cue-word-b__containsword__word': list(combinations(CUE_WORDS_WORSE, 3))
    }

    pipeline = GridSearchCV(pipeline, n_jobs=8, param_grid=param_grid, verbose=30, scoring=scorer, cv=2)

    fitted = pipeline.fit(train,
                          train['label'].values)
    predictions = fitted.predict(test)
    print("---------------------")
    print(pipeline.cv_results_)
    print("---------------------")
    print(pipeline.best_params_)
    print("---------------------")
    print(pipeline.best_score_)


run_grid_search(LinearSVC(), build_feature_union(_data), _data)

for estimator in [LogisticRegression(), LinearSVC(), RandomForestClassifier()]:
    pass
    # run_pipeline(estimator, build_feature_union(_data), _data, _labels)


def run_mlp(data, labels):
    print('****** MLP ******')
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    pipe = make_pipeline(ExtractRawSentence(), MeanWordEmbedding(),
                         MLPClassifier(verbose=True, warm_start=True, max_iter=1000, hidden_layer_sizes=(10, 5)))
    fitted = pipe.fit(train, train['label'].values)
    predictions = fitted.predict(test)

    report = classification_report(test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(test['label'].values, predictions, labels=labels)

    print(report)
    print(conf)

# run_mlp(_data, _labels)
#    print('######### BINARY #########')
#  run_pipeline(estimator, build_feature_union(_data), _data_bin, _labels_bin)
