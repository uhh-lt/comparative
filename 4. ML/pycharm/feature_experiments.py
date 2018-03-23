from pprint import pprint

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from xgboost import XGBClassifier

from classification_report_util import get_std_derivations, get_best_fold, latex_classification_report
from features.contains_features import ContainsPos
from features.mean_embedding_feature import MeanWordEmbedding
from features.ngram_feature import NGramFeature
from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractRawSentence, ExtractMiddlePart
from transformers.n_gram_transformers import NGramTransformer
from util.data_utils import load_data, k_folds
from util.misc_utils import get_logger
from util.ngram_utils import get_all_ngrams

ALL_EXTRACTORS = [('full sentence', ExtractRawSentence()),
                  ('full sentence replace', ExtractRawSentence(processing='replace')),
                  ('full sentence remove', ExtractRawSentence(processing='remove')),
                  ('full sentence remove dist', ExtractRawSentence(processing='remove_dist')),
                  ('middle part', ExtractMiddlePart()),
                  ('middle part replace', ExtractMiddlePart(processing='replace')),
                  ('middle part remove', ExtractMiddlePart(processing='remove')),
                  ('middle part remove dist', ExtractMiddlePart(processing='remove_dist')), ]


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


def all_extractor_combis(feature_class, name, *args):
    return [('{} - {}'.format(name, e[0]), FeatureUnion([(name, Pipeline([e, (name, feature_class(*args))]))])) for e
            in
            ALL_EXTRACTORS]


def all_extractor_ngram(n, base_ngrams, min_freq=1, filter_punct=True):
    return [('n-grams (n={}, min freq={}, filter punct={}) - {}'.format(n, min_freq, filter_punct, e[0]),
             FeatureUnion([(str(n), Pipeline(
                 [e, ('transformer', NGramTransformer(n=n, min_freq=min_freq, filter_punct=filter_punct)),
                  ('feat', NGramFeature(base_ngrams, n=n))]))])) for
            e
            in
            ALL_EXTRACTORS]


nlp = spacy.load('en')

logger = get_logger('feature_tests')
classifier = XGBClassifier(n_jobs=8, n_estimators=100)
LABEL = 'most_frequent_label'
data = load_data('data.csv')[:150]

infersent_model = initialize_infersent(data.sentence.values)
unigrams = get_all_ngrams(data.sentence.values, 1)
bigrams = get_all_ngrams(data.sentence.values, 2)
trigrams = get_all_ngrams(data.sentence.values, 3)

folds = list(k_folds(5, data))

feature_unions = [

                 ] \
                 + all_extractor_ngram(1, unigrams) \
                 + all_extractor_ngram(1, unigrams, filter_punct=False) \
                 + all_extractor_ngram(1, unigrams, min_freq=2) \
                 + all_extractor_ngram(1, unigrams, min_freq=10) \
                 + all_extractor_ngram(1, unigrams, filter_punct=False, min_freq=2) \
                 + all_extractor_ngram(2, bigrams) \
                 + all_extractor_ngram(2, bigrams, filter_punct=False) \
                 + all_extractor_ngram(2, bigrams, min_freq=2) \
                 + all_extractor_ngram(2, bigrams, filter_punct=False, min_freq=2) \
                 + all_extractor_ngram(3, trigrams) \
                 + all_extractor_ngram(3, trigrams, filter_punct=False) \
                 + all_extractor_ngram(3, trigrams, min_freq=2) \
                 + all_extractor_ngram(3, trigrams, filter_punct=False, min_freq=2) \
                 + all_extractor_combis(TfidfVectorizer, 'tfidf') \
                 + all_extractor_combis(InfersentFeature, 'Infersent', infersent_model) \
                 + all_extractor_combis(MeanWordEmbedding, 'Mean WordEmbedding') \
                 + all_extractor_combis(ContainsPos, 'Contains JJR', 'JJR') \
                 + all_extractor_combis(ContainsPos, 'Contains JJS', 'JJS') \
                 + all_extractor_combis(ContainsPos, 'Contains RBR', 'RBR') \
                 + all_extractor_combis(ContainsPos, 'Contains RBS', 'RBS')

best_per_feat = []
for caption, feature_union in feature_unions:
    logger.info(caption)
    folds_results = []
    for train, test in folds:
        pipeline = make_pipeline(feature_union, classifier)

        fitted = pipeline.fit(train, train[LABEL].values)
        predicted = fitted.predict(test)
        folds_results.append((test[LABEL].values, predicted))
        logger.info(classification_report(test[LABEL].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
    der = get_std_derivations(folds_results, ['BETTER', 'WORSE', 'NONE'])
    best = get_best_fold(folds_results)
    best_per_feat.append((f1_score(best[0], best[1], average='weighted'), caption))
    pprint(sorted(best_per_feat, key=lambda k: k[0], reverse=True))
    logger.info(latex_classification_report(best[0], best[1], derivations=der, labels=['BETTER', 'WORSE', 'NONE'],
                                            caption=caption))
    logger.info("\n\n=================\n\n")

logger.info(sorted(best_per_feat, key=lambda k: k[0], reverse=True))
