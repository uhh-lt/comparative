from collections import defaultdict
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import make_pipeline, FeatureUnion
from xgboost import XGBClassifier

from classification_report_util import get_std_derivations, get_best_fold, latex_classification_report
from features.contains_features import ContainsPos
from features.mean_embedding_feature import MeanWordEmbedding
from features.misc_features import SelectDataFrameColumn
from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractMiddlePart
from util.data_utils import load_data, k_folds, get_misclassified
from util.graphic_utils import print_confusion_matrix, plot
from util.misc_utils import get_logger
import json


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


nlp = spacy.load('en_core_web_lg')
logger = get_logger('finale_fassung')

LABEL = 'most_frequent_label'
data = load_data('data_if.csv')
data_bin = load_data('data_if.csv', binary=True)

# infersent_model = initialize_infersent(data.sentence.values)


best_per_feat = []


def perform_classificiation(data, labels):
    result_frame = pd.DataFrame(columns=['feature', 'class', 'f1', 'precision', 'recall'])
    conf_dict = defaultdict(lambda: np.zeros((len(labels), len(labels)), dtype=np.integer))

    feature_unions = [
        #  FeatureUnion([('Bag-Of-Words', make_pipeline(ExtractMiddlePart(), CountVectorizer()))]),
        FeatureUnion([('InferSent', make_pipeline(SelectDataFrameColumn('embedding_middle_part', value_transform=lambda x: json.loads(x))))]),
        #  FeatureUnion([('InferSent', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model)))]),
        #  FeatureUnion([('Word Embedding', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding()))]),
        #  FeatureUnion([('POS n-grams', make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))), ]),
        #   FeatureUnion([('Contains JJR', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR')))]),
    ]
    miss = pd.DataFrame(columns=['caption', 'sentence', 'object_a', 'object_b', 'predicted', 'gold'])
    binary = labels == ['ARG', 'NONE']
    idx_mc = 1
    idx = 1
    logger.info("====== {} =====".format(labels))
    for i, f in enumerate(feature_unions):
        caption = f.transformer_list[0][0]
        logger.info('{}/{} {}'.format(i, len(feature_unions), caption))
        logger.info(f)
        folds_results = []
        try:
            for train, test in k_folds(5, data, random_state=42):
                pipeline = make_pipeline(f, XGBClassifier(n_jobs=8, n_estimators=1000))
                fitted = pipeline.fit(train, train[LABEL].values)
                predicted = fitted.predict(test)
                folds_results.append((test[LABEL].values, predicted))
                logger.info(
                    classification_report(test[LABEL].values, predicted, labels=labels, digits=2))
                matrix = confusion_matrix(test[LABEL].values, predicted, labels=labels)
                logger.info(matrix)
                conf_dict[caption] += matrix

                result_frame.loc[idx] = [caption, 'Overall', f1_score(test[LABEL].values, predicted, average='weighted'), precision_score(test[LABEL].values, predicted, average='weighted'),
                                         recall_score(test[LABEL].values, predicted, average='weighted')]
                idx += 1
                for label in labels:
                    result_frame.loc[idx] = [caption, label, f1_score(test[LABEL].values, predicted, average='weighted', labels=[label]),
                                             precision_score(test[LABEL].values, predicted, average='weighted', labels=[label]),
                                             recall_score(test[LABEL].values, predicted, average='weighted', labels=[label])]
                    idx += 1

                for sentence, a, b, predicted, gold in get_misclassified(predicted, test):
                    miss.loc[idx_mc] = [caption, sentence, a, b, predicted, gold]
                    idx_mc += 1

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
        logger.info(conf_dict[caption])
        print_confusion_matrix('{}_{}'.format(caption, binary), conf_dict[caption], labels)
        logger.info("\n\n=================\n\n")
    logger.info(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
    miss.to_csv('missclassified/binary_{}.csv'.format(binary), index=False)
    result_frame.to_csv('graphics/data/results_{}.csv'.format(binary), index=False)
    plot(result_frame)


perform_classificiation(data, ['BETTER', 'WORSE', 'NONE'])
perform_classificiation(data_bin, ['ARG', 'NONE'])