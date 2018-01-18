import time
from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from multiprocessing.pool import ThreadPool
from features.counts import *
from features.ngram_feature import NGramFeature
from transformers.data_extraction import *
from transformers.n_gram_transformers import NGramTransformer
from util.feature_builder import *
from util.ngram import get_all_ngrams

MIN_F1 = 0.72

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
    return shuffle(frame)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits, random_state=42)
    for train_index, test_index in k_fold.split(data,
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def setup_n_grams(d):
    return get_all_ngrams(d['raw_text'].values, n=1), get_all_ngrams(d['raw_text'].values, n=2)


def build_features(data):
    """ feat = [

         ('vsm', make_pipeline(ExtractRawSentence(), TfidfVectorizer())),
         ('vsm-l', make_pipeline(ExtractLastPart(), TfidfVectorizer())),
         ('vsm-m', make_pipeline(ExtractMiddlePart(), TfidfVectorizer())),
         ('vsm-f', make_pipeline(ExtractFirstPart(), TfidfVectorizer())),

         ('unigram-l', make_pipeline(ExtractLastPart(), NGram(unigrams, n=1))),

         ('length-m', make_pipeline(ExtractMiddlePart(), Length())),
         ('length-f', make_pipeline(ExtractFirstPart(), Length())),
         ('length-l', make_pipeline(ExtractLastPart(), Length())),
         ('length', make_pipeline(ExtractRawSentence(), Length())),

         ('bigram-l', make_pipeline(ExtractLastPart(), NGram(bigrams, n=2))),
         ('bigram-m', make_pipeline(ExtractMiddlePart(), NGram(bigrams, n=2))),
         ('bigram-f', make_pipeline(ExtractFirstPart(), NGram(bigrams, n=2))),
         ('bigram', make_pipeline(ExtractRawSentence(), NGram(bigrams, n=2))),

         ('punct', make_pipeline(ExtractRawSentence(), PunctuationCount())),
         ('punct-m', make_pipeline(ExtractMiddlePart(), PunctuationCount())),
         ('punct-f', make_pipeline(ExtractFirstPart(), PunctuationCount())),
         ('punct-l', make_pipeline(ExtractLastPart(), PunctuationCount())),

         ('mwe-m', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding())),
         ('mwe', make_pipeline(ExtractRawSentence(), MeanWordEmbedding())),

         ('all-w', make_pipeline(ExtractRawSentence(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),
         ('all-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),
         ('all-f', make_pipeline(ExtractFirstPart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),

         ('ne', make_pipeline(ExtractRawSentence(), NECount())),
         ('ne-f', make_pipeline(ExtractFirstPart(), NECount())),
         ('ne-l', make_pipeline(ExtractLastPart(), NECount())),

         ('nc', make_pipeline(ExtractRawSentence(), NounChunkCount())),
         ('nc-f', make_pipeline(ExtractFirstPart(), NounChunkCount())),
         ('nc-l', make_pipeline(ExtractLastPart(), NounChunkCount()))
     ]

     combis = list(combinations(feat, 1))
     print('{} feature combinations'.format(len(combis)))"""
    return []


# ------ Classification


_labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
_labels_bin = ['ARG', 'NONE']

_data = load_data('train-data.csv')
_data_bin = load_data('train-data.csv', binary=True)

unigrams, bigrams = setup_n_grams(_data_bin)
"""
best_so_far = [
    ('unigram-m', make_pipeline(ExtractMiddlePart(), NGramFeature(unigrams, n=1))),  # 0.6878
    ('unigram-f', make_pipeline(ExtractFirstPart(), NGramFeature(unigrams, n=1))),  # 0.7050
    ('unigram', make_pipeline(ExtractRawSentence(), NGramFeature(unigrams, n=1))),  # 0.7113
    ('all-l', make_pipeline(ExtractLastPart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),  # 0.716
    ('ne-m', make_pipeline(ExtractMiddlePart(), NEOverallCount())),  # 0.7184,
    ('nc-m', make_pipeline(ExtractMiddlePart(), NounChunkCount())),  # Score 0.7200

]
"""
test_feat = [
    ('unigram', make_pipeline(ExtractMiddlePart(), NGramTransformer(), NGramFeature(unigrams)))
]


def run_pipeline(estimator, feature_union, train, test, labels):
    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])

    fitted = pipeline.fit(train,
                          train['label'].values)

    predictions = fitted.predict(test)

    report = classification_report(test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(test['label'].values, predictions, labels=labels)
    f1 = f1_score(test['label'].values, predictions, labels=labels, average='weighted')
    return f1, str(report), str(conf)


def run(data, label):
    for estimator in [
        LinearSVC()]:  # , DecisionTreeClassifier(), SGDClassifier(), LinearSVC(), NearestCentroid()]:
        run_estimator(estimator, data, label)


def run_estimator(estimator, data, label):
    key = str(type(estimator))
    print('Start {}'.format(key))
    f1_sum = 0
    for train, test in split_data(3, data):
        f1, report, conf = run_pipeline(estimator, FeatureUnion(test_feat), train, test, label)
        print(report)
        print(conf)
        f1_sum += f1
    print('\n{} Average F1 Score {}\n==========================\n\n'.format(key, f1_sum / 3))


run(_data, _labels)

# run(_data_bin, _labels_bin, 'lb')
