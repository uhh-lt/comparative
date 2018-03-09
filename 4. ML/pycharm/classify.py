from nltk import NaiveBayesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
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
from util.misc_utils import latex_table, get_logger
from util.ngram_utils import get_all_ngrams
from collections import OrderedDict
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier
import datetime

classifier_pattern = """
----------------------
- {}
----------------------"""
feature_name_pattern = """
========================================================================================

======================
- {}
======================"""

logger = get_logger('xgb_dr_wo')

classifiers = [XGBClassifier(n_jobs=4, n_estimators=25)]
# classifiers = [RidgeClassifier(), XGBClassifier(), LinearSVC(), SVC(), SGDClassifier(), GaussianNB(),
#               KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
#               LogisticRegression(), ExtraTreesClassifier()]

n_gram_cache = {}


def n_gram_pipeline(n, extractor, processing=None):
    def _n_gram_pipeline(train):
        ngram_base = extractor(processing=processing).transform(train)
        unigrams = get_all_ngrams(ngram_base, n)
        return [extractor(processing=processing), NGramTransformer(n), NGramFeature(unigrams, n=n)]

    return _n_gram_pipeline


def infersent_pipeline(extractor, processing=None):
    def _infersent_pipeline(train):
        raw_text = extractor(processing=processing).transform(train)
        infersent_model = initialize_infersent(raw_text)
        return [extractor(processing=processing), InfersentFeature(infersent_model)]

    return _infersent_pipeline


feature_builder = [

    # ('Contains JJR WS', lambda train: [ExtractRawSentence(), ContainsPos('JJR')]),

    # ('NE Count MP', lambda train: [ExtractMiddlePart(), NamedEntitiesByCategory()]),
    # ('Noun Chunk Count WS', lambda train: [ExtractRawSentence(), NounChunkCount()]),
    # ('Noun Chunk MP', lambda train: [ExtractMiddlePart(), NounChunkCount()]),
    # ('Position of Objects WS', lambda train: [PositionOfObjects()]),

    #  ('Bigram WS', n_gram_pipeline(2, ExtractRawSentence)),
    ('Bigram MP', n_gram_pipeline(2, ExtractMiddlePart)),
    ('Bigram MP + DR', n_gram_pipeline(2, ExtractMiddlePart, 'replace_dist')),
    # ('Trigram WS', n_gram_pipeline(3, ExtractRawSentence)),
    # ('Trigram MP', n_gram_pipeline(3, ExtractMiddlePart)),
    # ('Unigram WS', n_gram_pipeline(1, ExtractRawSentence)),
    ('Unigram MP', n_gram_pipeline(1, ExtractMiddlePart)),
    ('Unigram MP DR', n_gram_pipeline(1, ExtractMiddlePart, 'replace_dist')),
    # ('Unigram MP Remove', n_gram_pipeline(1, ExtractMiddlePart, 'remove')),

    # ('Sentence Embedding WS', infersent_pipeline(ExtractRawSentence)),
    ('Sentence Embedding MP', infersent_pipeline(ExtractMiddlePart)),
    ('Sentence Embedding MP + DR', infersent_pipeline(ExtractMiddlePart, 'replace_dist')),

    ('Mean Word Embedding MP', lambda train: [ExtractMiddlePart(), MeanWordEmbedding()]),
    ('Mean Word Embedding MP + DR', lambda train: [ExtractMiddlePart(processing='replace_dist'), MeanWordEmbedding()]),

    ('Contains JJR MP', lambda train: [ExtractMiddlePart(), ContainsPos('JJR')]),
    ('Contains JJR MP DR', lambda train: [ExtractMiddlePart(processing='replace_dist'), ContainsPos('JJR')]),
    #  ('Contains JJS MP', lambda train: [ExtractMiddlePart(), ContainsPos('JJS')]),
    ('Contains RBR MP', lambda train: [ExtractMiddlePart(), ContainsPos('RBR')]),
    ('Contains RBR MP + DR', lambda train: [ExtractMiddlePart(processing='replace_dist'), ContainsPos('RBR')]),
    # ('Contains RBS MP', lambda train: [ExtractMiddlePart(), ContainsPos('RBS')]),
    # # ('Punctuation Count WS', lambda train: [ExtractRawSentence(), PunctuationCount()]),
    # ('Punctuation Count MP', lambda train: [ExtractMiddlePart(), PunctuationCount()]),
    # ('NE Count WS', lambda train: [ExtractRawSentence(), NamedEntitiesByCategory()]),

]


def run_classification(data, labels):
    by_score = []
    for _builder in feature_builder:
        name, builder = _builder
        logger.info(feature_name_pattern.format(name.upper()))
        for classifier in classifiers:

            logger.info(classifier_pattern.format(classifier))

            res = []
            f1_overall = 0;
            for train, test in k_folds(5, data):
                try:
                    steps = builder(train) + [classifier]
                    pipeline = make_pipeline(*steps)
                    fitted = pipeline.fit(train, train['label'].values)
                    predicted = fitted.predict(test)
                    logger.info(classification_report(test['label'].values, predicted, labels=labels))
                    f1 = f1_score(test['label'].values, predicted, average='weighted',
                                  labels=labels)
                    f1_overall += f1
                    print(steps)

                    res.append((f1_score(test['label'].values, predicted, average='weighted',
                                         labels=labels), (test['label'].values, predicted)))
                    now = datetime.datetime.now()
                    logger.info("{}:{}:{}\n\n".format(now.hour, now.minute, now.second))
                except Exception as e:
                    logger.info(e)
                    logger.info('Fail for {}'.format(type(classifier)).upper())
            res = sorted(res, key=lambda x: x[0])
            logger.info('OVERALL F1 {}'.format(f1_overall / 5.0))
            by_score.append((f1_overall / 5.0, '{} {}'.format(type(classifier), name)))
            try:
                logger.info(latex_table([res[0][1]] + [res[2][1]] + [res[4][1]], 'cap'))
            except Exception as e:
                logger.info("No table")
    logger.info(sorted(by_score, key=lambda x: x[0], reverse=True))


logger.info('# THREE CLASSES')
_data = load_data('data.csv', min_confidence=0, binary=False)
run_classification(_data, ['BETTER', 'WORSE', 'NONE'])
logger.info('\n\n--------------------------------------------\n\n')
logger.info('# BINARY CLASSES')
_data_bin = load_data('data.csv', min_confidence=0, binary=True)

# run_classification(_data_bin, ['ARG', 'NONE'])
