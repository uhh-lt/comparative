import pandas as pd
import spacy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
from textacy.extract import subject_verb_object_triples
from textacy.spacy_utils import is_negated_verb
from xgboost import XGBClassifier
from pprint import pprint
import util.data_utils as du
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from features.contains_features import ContainsWord
from features.misc_features import SelectAllPaths, PositionOfWord, MeanPathVector
from infersent.infersent_feature import initialize_infersent, InfersentFeature
from transformers.data_extraction import ExtractMiddlePart, ExtractRawSentence
from util.misc_utils import get_logger

nlp = spacy.load('en')


class NegationFeature(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, sentence):
        feat = []
        for s in sentence:
            b = list(subject_verb_object_triples(nlp(s)))
            a = list(nlp(s).sents)[0]
            p = sum([1 if is_negated_verb(t) else 0 for t in a])
            feat.append(p)

        return np.array(feat).reshape(-1, 1)


LABEL = 'most_frequent_label'
data = du.load_data('data.csv')
paths = pd.read_csv('data/dependency_paths.csv')

paths_vocab = list(paths.path)
vectorizer = CountVectorizer(vocabulary=paths_vocab)
w2v = Word2Vec.load('data/dependency_w2v_gensim')

logger = get_logger('dependency_tests')
classifier = XGBClassifier(n_jobs=8, n_estimators=100)

# model = initialize_infersent(data.sentence.values.tolist())


for train, test in du.k_folds(5, data):
    pipeline = make_pipeline(FeatureUnion([
        ('uni', make_pipeline(MeanPathVector(w2v,paths)))
    ]),
        classifier)

    fitted = pipeline.fit(train, train[LABEL].values)
    predicted = fitted.predict(test)
    logger.info(
        classification_report(test[LABEL].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
