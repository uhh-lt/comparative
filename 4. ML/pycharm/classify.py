from itertools import combinations

from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from time import gmtime, strftime
from features.counts import *
from features.mean_wordembedding import MeanWordEmbedding
from transformers.data_extraction import *
from util.feature_builder import *
from util.ngram import get_all_ngrams

MIN_F1 = 0.55

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
    return shuffle(frame, random_state=42)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits)
    for train_index, test_index in k_fold.split(data['raw_text'],
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def setup_n_grams(d):
    _unigrams = get_all_ngrams(d['raw_text'].values, n=1)
    _bigrams = get_all_ngrams(d['raw_text'].values, n=2)
    print('build ngram dicts {} {}'.format(len(_unigrams), len(_bigrams)))
    return _unigrams, _bigrams


def build_features(data):
    feat = [
        ('worse-w', make_pipeline(ExtractRawSentence(), ContainsWord(CUE_WORDS_WORSE))),
        ('worse-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_WORSE))),
        ('worse-l', make_pipeline(ExtractLastPart(), ContainsWord(CUE_WORDS_WORSE))),
        ('worse-f', make_pipeline(ExtractFirstPart(), ContainsWord(CUE_WORDS_WORSE))),

        ('better-w', make_pipeline(ExtractRawSentence(), ContainsWord(CUE_WORDS_BETTER))),
        ('better-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER))),
        ('better-l', make_pipeline(ExtractLastPart(), ContainsWord(CUE_WORDS_BETTER))),
        ('better-f', make_pipeline(ExtractFirstPart(), ContainsWord(CUE_WORDS_BETTER))),

        ('all-w', make_pipeline(ExtractRawSentence(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),
        ('all-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),
        ('all-l', make_pipeline(ExtractLastPart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),
        ('all-f', make_pipeline(ExtractFirstPart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),
    ]
    combis = list(combinations(feat, 1)) + list(combinations(feat, 2)) + list(combinations(feat, 3))

    print('{} feature combinations'.format(len(combis)))
    return combis

    # ------ Classification


_labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
_labels_bin = ['ARG', 'NONE']

_data = load_data('train-data.csv')
_data_bin = load_data('train-data.csv', binary=True)

unigrams, bigrams = setup_n_grams(_data_bin)


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
    f1 = f1_score(test['label'].values, predictions, labels=labels, average='weighted')
    if f1 > MIN_F1:
        print('F1: {}'.format(f1))
        print(report)
        print(conf)
        return f1, str(report), str(conf)
    else:
        print('F1: {}'.format(f1))
        return f1, '-', '-'


def run(d, l, fn):
    features = build_features(d);
    with open('{}.log'.format(fn), 'w') as log:
        counter = 0

        log.write("FEATURES:\n")
        log.write(str(features))
        for feat_combi in features:
            for estimator in [LinearSVC(), SGDClassifier()]:
                print(str(counter))
                counter += 1
                print(feat_combi)

                f1, report, conf = run_pipeline(estimator, FeatureUnion(list(feat_combi)), d, l)
                if f1 > MIN_F1:
                    log.write(str(type(estimator)))
                    log.write('\n')
                    log.write(str(feat_combi))
                    log.write('\n')
                    log.write(report)
                    log.write('\n')
                    log.write(conf)
                    log.write("\n========================\n\n")
                print('--------------\n\n')


run(_data, _labels, 'marker')
# run(_data_bin, _labels_bin)
