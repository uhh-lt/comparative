import itertools
from pprint import pformat

import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from xgboost import XGBClassifier

from classification_report_util import get_std_derivations, get_best_fold, latex_classification_report
from features.contains_features import ContainsPos
from features.mean_embedding_feature import MeanWordEmbedding
from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractRawSentence, ExtractMiddlePart
from util.data_utils import load_data, k_folds, get_misclassified
from util.misc_utils import get_logger


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
                  ('middle part', ExtractMiddlePart()),
                  ('middle part replace', ExtractMiddlePart(processing='replace')),
                  ('middle part remove', ExtractMiddlePart(processing='remove')),
                  ('middle part remove dist', ExtractMiddlePart(processing='remove_dist'))]


def all_extractor_combis(feature_class, name, *args):
    return [FeatureUnion([('{} {}'.format(name,e), Pipeline([e, (name, feature_class(*args))]))]) for e in ALL_EXTRACTORS]


nlp = spacy.load('en')

logger = get_logger('final_1')
classifier = XGBClassifier(n_jobs=8, n_estimators=100)
LABEL = 'most_frequent_label'
data = load_data('data.csv')[:150]
data_bin = load_data('data.csv', binary=True)

infersent_model = initialize_infersent(data.sentence.values)

feature_unions = [
                     FeatureUnion([('2-4 pos 500 full sentence', make_pipeline(ExtractRawSentence(), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))), ]),
                     FeatureUnion([('2-4 pos 500 middle part', make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))), ]),
                     FeatureUnion([('2-4 pos 500 middle part replace', make_pipeline(ExtractMiddlePart(processing='replace'), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))), ]),
                     FeatureUnion([('2-4 pos 500 middle part remove', make_pipeline(ExtractMiddlePart(processing='remove'), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))), ]),
                     FeatureUnion([('2-4 pos 500 middle part replace_dist', make_pipeline(ExtractMiddlePart(processing='replace_dist'), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))), ]),

                 ] + all_extractor_combis(InfersentFeature, 'infersent', infersent_model) + all_extractor_combis(MeanWordEmbedding, 'Mean Word Embedding') + all_extractor_combis(ContainsPos, 'Contains JJR', 'JJR')

best_per_feat = []


def perform_classificiation(data, labels):
    miss = pd.DataFrame(columns=['caption', 'sentence', 'object_a', 'object_b', 'predicted', 'gold'])
    binary = labels == ['ARG', 'NONE']
    idx = 1
    logger.info("====== {} =====".format(labels))
    for i, f in enumerate(feature_unions):
        caption = f.transformer_list[0][0]
        logger.info('{}/{} {}'.format(i, len(feature_unions), caption))
        logger.info(f)
        folds_results = []
        try:
            for train, test in k_folds(5, data, random_state=1337):
                pipeline = make_pipeline(f, classifier)

                fitted = pipeline.fit(train, train[LABEL].values)
                predicted = fitted.predict(test)
                folds_results.append((test[LABEL].values, predicted))
                logger.info(
                    classification_report(test[LABEL].values, predicted, labels=labels, digits=2))

                logger.info(confusion_matrix(test[LABEL].values, predicted, labels=labels))

                for sentence, a, b, predicted, gold in get_misclassified(predicted, test):
                    miss.loc[idx] = [caption, sentence, a, b, predicted, gold]
                    idx += 1

            der = get_std_derivations(folds_results, labels=labels)
            best = get_best_fold(folds_results)
            best_per_feat.append((f1_score(best[0], best[1], average='weighted'), caption))
            print(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
            logger.info(
                latex_classification_report(best[0], best[1], derivations=der, labels=labels,
                                            caption=caption))

        except Exception as ex:
            logger.error(ex)
            raise ex
        logger.info("\n\n=================\n\n")
    logger.info(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
    miss.to_csv('missclassified/binary_{}.csv'.format(binary), index=False)


perform_classificiation(data, ['BETTER', 'WORSE', 'NONE'])
perform_classificiation(data_bin, ['ARG', 'NONE'])
