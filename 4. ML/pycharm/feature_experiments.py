import itertools

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from xgboost import XGBClassifier

from classification_report_util import get_std_derivations, get_best_fold, latex_classification_report
from features.contains_features import ContainsPos
from features.mean_embedding_feature import MeanWordEmbedding
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
    ranges = [(2, 2), (3, 3), (4, 4), (2, 4)]
    binary = [True, False]
    top_k = [None, 100]
    feat = []
    for c in itertools.product(ranges, binary, top_k):
        feat += ([('{} Range {} Binary {} Top {} ({} {})'.format(e[0], c[0], c[1], c[2], type(vectorizer()), name_add),
                   FeatureUnion([(
                       '{} {}'.format(
                           e, c),
                       Pipeline(
                           [e, ('{} {}'.format(e, c),
                                vectorizer(
                                    ngram_range=c[0], binary=c[1], max_features=c[2], **kwargs))]))]))
                  for e in ALL_EXTRACTORS])
    return feat


nlp = spacy.load('en')

logger = get_logger('feature_tests_pos_gram')
classifier = XGBClassifier(n_jobs=8, n_estimators=100)
LABEL = 'most_frequent_label'
data = load_data('data.csv')[:50]

infersent_model = initialize_infersent(data.sentence.values)

folds = k_folds(5,data,random_state=1337)



features = [
    ('infersent middle', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model))),
    ('mean word embedding middle', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding())),
    ('unigram counts binary top 100',
     make_pipeline(ExtractMiddlePart(), CountVectorizer(binary=True, max_features=100))),
    ('unigram counts binary all', make_pipeline(ExtractMiddlePart(), CountVectorizer(binary=True))),
    ('unigram tfidf all', make_pipeline(ExtractMiddlePart(), TfidfVectorizer())),
    ('unigram tfidf top 100', make_pipeline(ExtractMiddlePart(), TfidfVectorizer(max_features=100))),
    ('contains jjr', make_pipeline(ExtractMiddlePart(), ContainsPos("JJR"))),
    ('2-4 pos 100',
     make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(max_features=100, ngram_range=(2, 4)))),
    ('2-4 pos 500',
     make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))),
    ('2-4 pos all', make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(ngram_range=(2, 4)))),
    ('2 pos all', make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(ngram_range=(2, 2)))),
    ('2 pos 100',
     make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(ngram_range=(2, 2), max_features=100))),
    ('word vector', make_pipeline(WordVector()))
]

feature_unions = []
combis = list(itertools.combinations(features,1)) + list(itertools.combinations(features,2)) + list(itertools.combinations(features,3))
for c in combis:
    n = ' | '.join([a[0] for a in c])
    feature_unions.append((n, FeatureUnion(transformer_list=list(c))))


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
