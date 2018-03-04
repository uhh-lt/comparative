from nltk import NaiveBayesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from features.contains_features import ContainsPos
from features.count_features import PunctuationCount, NamedEntitiesByCategory, NounChunkCount
from features.mean_embedding_feature import MeanWordEmbedding
from features.misc_features import PositionOfObjects
from features.ngram_feature import NGramFeature
from infersent.infersent_feature import initialize_infersent, InfersentFeature
from transformers.data_extraction import ExtractRawSentence, ExtractMiddlePart
from transformers.n_gram_transformers import NGramTransformer
from util.data_utils import load_data, k_folds
from util.misc_utils import latex_table
from util.ngram_utils import get_all_ngrams
from collections import OrderedDict
from pprint import pprint

import datetime

now = datetime.datetime.now()

classifier_pattern = """
----------------------
- {}
----------------------"""
feature_name_pattern = """
========================================================================================

======================
- {}
======================"""

classifiers = [LinearSVC(), SGDClassifier(), GaussianNB(), KNeighborsClassifier(),
               SVC(), RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(),
               LogisticRegression(), XGBClassifier()]


def n_gram_pipeline(n, extractor, processing=None):
    def _n_gram_pipeline(train):
        ngram_base = extractor(processing=processing).transform(train)
        unigrams = get_all_ngrams(ngram_base, n)
        return [extractor(processing=processing), NGramTransformer(n), NGramFeature(unigrams)]

    return _n_gram_pipeline


def infersent_pipeline(extractor, processing=None):
    def _infersent_pipeline(train):
        raw_text = extractor(processing=processing).transform(train)
        infersent_model = initialize_infersent(raw_text)
        return [extractor(processing=processing), InfersentFeature(infersent_model)]

    return _infersent_pipeline


feature_builder = [

    ('Mean Word Embedding WS', lambda train: [ExtractRawSentence(), MeanWordEmbedding()]),
    ('Unigram WS', n_gram_pipeline(1, ExtractRawSentence)),
    (
        'Unigram WS + DR',
        n_gram_pipeline(1, ExtractRawSentence, 'replace_dist')),
    ('Unigram MP', n_gram_pipeline(1, ExtractMiddlePart)),
    (
        'Unigram WS + DR',
        n_gram_pipeline(1, ExtractMiddlePart, 'replace_dist')), (
        'Unigram WS + RE',
        n_gram_pipeline(1, ExtractMiddlePart, 'remove')),
    ('Bigram WS', n_gram_pipeline(2, ExtractRawSentence)),
    ('Bigram MP', n_gram_pipeline(2, ExtractMiddlePart)),
    ('Trigram WS', n_gram_pipeline(3, ExtractRawSentence)),
    ('Trigram MP', n_gram_pipeline(3, ExtractMiddlePart)),
    ('Contains JJR WS', lambda train: [ExtractRawSentence(), ContainsPos('JJR')]),
    ('Contains JJR MP', lambda train: [ExtractMiddlePart(), ContainsPos('JJR')]),

    ('Punctuation Count WS', lambda train: [ExtractRawSentence(), PunctuationCount()]),
    ('Punctuation Count MP', lambda train: [ExtractMiddlePart(), PunctuationCount()]),
    ('NE Count WS', lambda train: [ExtractRawSentence(), NamedEntitiesByCategory()]),
    ('NE Count MP', lambda train: [ExtractMiddlePart(), NamedEntitiesByCategory()]),
    ('Noun Chunk Count WS', lambda train: [ExtractRawSentence(), NounChunkCount()]),
    ('Noun Chunk MP', lambda train: [ExtractMiddlePart(), NounChunkCount()]),
    ('Position of Objects WS', lambda train: [PositionOfObjects()]),
    ('Sentence Embedding WS', infersent_pipeline(ExtractRawSentence)),
    ('Sentence Embedding MP', infersent_pipeline(ExtractMiddlePart)),
    ('Sentence Embedding WS + DR', infersent_pipeline(ExtractRawSentence, 'replace_dist')),
    ('Sentence Embedding MP + DR', infersent_pipeline(ExtractMiddlePart, 'replace_dist')),

]


def run_classification(data, labels):
    by_score = []
    for _builder in feature_builder:
        name, builder = _builder
        print(feature_name_pattern.format(name.upper()))
        for classifier in classifiers:

            print(classifier_pattern.format(classifier))

            res = []

            for train, test in k_folds(5, data):
                steps = builder(train) + [classifier]
                pipeline = make_pipeline(*steps)
                fitted = pipeline.fit(train, train['label'].values)
                predicted = fitted.predict(test)
                print(classification_report(test['label'].values, predicted, labels=labels))
                f1 = f1_score(test['label'].values, predicted, average='weighted',
                              labels=labels)
                by_score.append((f1, '{} {}'.format(type(classifier), name)))

                res.append((f1_score(test['label'].values, predicted, average='weighted',
                                     labels=labels), (test['label'].values, predicted)))

                print("\n\n")
            res = sorted(res, key=lambda x: x[0])
            # latex_table([res[0][1]] + [res[2][1]] + [res[4][1]], 'cap')
    pprint(sorted(by_score, key=lambda x: x[0], reverse=True))


print('# THREE CLASSES')
_data = load_data('data.csv', min_confidence=0, binary=False)[:10]
run_classification(_data, ['BETTER', 'WORSE', 'NONE'])
print('\n\n--------------------------------------------\n\n')
print('# BINARY CLASSES')
_data_bin = load_data('data.csv', min_confidence=0, binary=True)

# run_classification(_data_bin, ['ARG', 'NONE'])
