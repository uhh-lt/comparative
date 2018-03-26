import itertools

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from xgboost import XGBClassifier

from classification_report_util import get_std_derivations, get_best_fold, latex_classification_report
from features.contains_features import ContainsPos
from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractRawSentence, ExtractMiddlePart, ExtractFirstPart, ExtractLastPart
from util.data_utils import load_data, k_folds
from util.misc_utils import get_logger
from pprint import pformat


class WordVector(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        feat = []
        for i, row in df.iterrows():
            a = list(nlp(row['object_a']).sents)[0][0].vector
            b = list(nlp(row['object_b']).sents)[0][0].vector
            feat.append(np.concatenate((a, b)))

        return feat


class POSTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, sentence):
        feat = []
        for s in sentence:
            a = list(nlp(s).sents)[0]
            p = ' '.join([t.pos_ for t in a])
            feat.append(p)

        return feat


ALL_EXTRACTORS = [('full sentence', ExtractRawSentence()),
                  ('full sentence replace', ExtractRawSentence(processing='replace')),
                  ('full sentence remove', ExtractRawSentence(processing='remove')),
                  ('full sentence remove dist', ExtractRawSentence(processing='remove_dist')),
                  ('middle part', ExtractMiddlePart()),
                  ('middle part replace', ExtractMiddlePart(processing='replace')),
                  ('middle part remove', ExtractMiddlePart(processing='remove')),
                  ('middle part remove dist', ExtractMiddlePart(processing='remove_dist'))]


def all_extractor_combis(feature_class, name, *args):
    return [('{} - {}'.format(name, e[0]), FeatureUnion([(name, Pipeline([e, (name, feature_class(*args))]))])) for e
            in
            ALL_EXTRACTORS]


def n_gram(vectorizer, name_add='', **kwargs):
    ranges = [(1, 1), (2, 2), (3, 3), (1, 3)]
    binary = [True, False]
    top_k = [None, 100, 2500]
    feat = []
    for c in itertools.product(ranges, binary, top_k):
        feat += ([('{} Range {} Binary {} Top {} ({})'.format(e[0], c[0], c[1], c[2], name_add), FeatureUnion([(
            '{} {}'.format(
                e,
                c),
            Pipeline(
                [e, (
                    '{} {}'.format(
                        e,
                        c),
                    vectorizer(
                        ngram_range=
                        c[
                            0],
                        binary=
                        c[
                            1],
                        max_features=
                        c[
                            2],
                        **kwargs))]))]))
                  for e
                  in
                  ALL_EXTRACTORS])
    return feat


nlp = spacy.load('en')

logger = get_logger('feature_tests')
classifier = XGBClassifier(n_jobs=8, n_estimators=100)
LABEL = 'most_frequent_label'
data = load_data('data.csv')

infersent_model = initialize_infersent(data.sentence.values)
# unigrams = get_all_ngrams(data.sentence.values, 1)
# bigrams = get_all_ngrams(data.sentence.values, 2)
# trigrams = get_all_ngrams(data.sentence.values, 3)

# pos_bigrams = get_all_ngrams(POSTransformer().transform(data.sentence.values), 2)

# pos_trigrams = get_all_ngrams(POSTransformer().transform(data.sentence.values), 3)
# pos_trigrams_mf_5 = get_all_ngrams(POSTransformer().transform(data.sentence.values), n=3, min_freq=5)

folds = list(k_folds(5, data))

feature_unions = [

                     ('pos bigrams middle + contains jrr middle ', FeatureUnion([
                         ('pos bigrams middle',
                          make_pipeline(ExtractMiddlePart(), CountVectorizer(ngram_range=(2, 2), binary=True))),
                         ('contains jrr middle', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR')))
                     ])),

                     ('pos bigrams full sentence + infersent middle',
                      FeatureUnion([
                          ('pos bigram',
                           make_pipeline(ExtractRawSentence(), POSTransformer(),
                                         CountVectorizer(ngram_range=(2, 2), binary=True))),
                          ('infersent', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model)))
                      ])),

                     ('unigram first + infersent middle + unigram last',
                      FeatureUnion([
                          ('unigram first',
                           make_pipeline(ExtractFirstPart(), CountVectorizer(binary=True))),
                          ('infersent', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model))),
                          ('unigram last',
                           make_pipeline(ExtractLastPart(), CountVectorizer(binary=True)))
                      ])),

                     ('unigrams min freq 2 full + infersent middle',
                      FeatureUnion([
                          ('unigrams',
                           make_pipeline(ExtractRawSentence(), CountVectorizer(min_df=2))),
                          ('infersent', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model)))
                      ])),

                     ('infersent first + infersent middle + inferset last',
                      FeatureUnion([
                          ('infersent first', make_pipeline(ExtractFirstPart(), InfersentFeature(infersent_model))),
                          ('infersent middle', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model))),
                          ('infersent last', make_pipeline(ExtractLastPart(), InfersentFeature(infersent_model)))
                      ])),

                     ('tfidf middle + infersent middle',
                      FeatureUnion([
                          ('infersent middle', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model))),
                          ('tfidf middle', make_pipeline(ExtractMiddlePart(), TfidfVectorizer())),
                      ])),

                 ] + n_gram(CountVectorizer) + n_gram(TfidfVectorizer) + n_gram(TfidfVectorizer, name_add='smoothed',
                                                                                smooth_idf=True)

best_per_feat = []
for caption, feature_union in feature_unions:
    logger.info(caption)
    logger.info(feature_union)
    folds_results = []
    try:
        for train, test in folds:
            pipeline = make_pipeline(feature_union, classifier)

            fitted = pipeline.fit(train, train[LABEL].values)
            predicted = fitted.predict(test)
            folds_results.append((test[LABEL].values, predicted))
            logger.info(
                classification_report(test[LABEL].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
        der = get_std_derivations(folds_results, ['BETTER', 'WORSE', 'NONE'])
        best = get_best_fold(folds_results)
        best_per_feat.append((f1_score(best[0], best[1], average='weighted'), caption))
        print(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
        logger.info(latex_classification_report(best[0], best[1], derivations=der, labels=['BETTER', 'WORSE', 'NONE'],
                                                caption=caption))
    except Exception as ex:
        logger.error(ex)
    logger.info("\n\n=================\n\n")

logger.info(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
