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
from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractMiddlePart
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


nlp = spacy.load('en_core_web_lg')
sns.set(font_scale=1.5, style="whitegrid")
logger = get_logger('bin')

LABEL = 'most_frequent_label'
data = load_data('data.csv')
data_bin = load_data('data.csv', binary=True)

infersent_model = initialize_infersent(data.sentence.values)

best_per_feat = []


def print_confusion_matrix(name, confusion_matrix, class_names):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names)
    df_cm.to_csv('graphics/data/conf-{}.csv'.format(name))
    fig = plt.figure(figsize=(8, 8))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", linewidths=1,
                              cmap="Greens", cbar=False)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=90, ha='center', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    heatmap.yaxis.set_label_coords(-0.1, 0.5)
    heatmap.xaxis.set_label_coords(0.5, -0.1)
    fig.savefig('graphics/conf-{}.pdf'.format(name))


def plot(d):
    for p in ['f1','precision','recall']:
        pal = sns.color_palette("muted")[:2]+[sns.color_palette("muted")[3]] if (len(d['class'].unique()) == 3) else  sns.color_palette("muted")
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        sns.barplot(x="feature", y=p, ci="sd",hue="class",palette=pal, data=d,dodge=True)
        plt.ylim(ymax = 1,ymin=0)
        plt.legend(ncol=4,loc='upper center', bbox_to_anchor=(0.5, 1.08))
        #plt.title("recall")
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(np.arange(0.0, 1.05, 0.05),minor=True)
        ax.yaxis.grid(which='minor', linestyle='--')
        ax.yaxis.grid(which='major', linestyle='-')
        plt.xlabel('')
        plt.ylabel('')
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=15)
        fig.savefig("graphics/{}-{}.pdf".format(p,len(d['class'].unique()) == 3))


def perform_classificiation(data, labels):
    result_frame = pd.DataFrame(columns=['feature', 'class', 'f1', 'precision', 'recall'])
    conf_dict = defaultdict(lambda: np.zeros((len(labels), len(labels)), dtype=np.integer))

    feature_unions = [
        FeatureUnion([('Unigrams', make_pipeline(ExtractMiddlePart(), CountVectorizer()))]),
        FeatureUnion([('InferSent', make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model)))]),
        FeatureUnion([('Word Embedding', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding()))]),
        FeatureUnion([('POS n-grams', make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4)))), ]),
        FeatureUnion([('Contains JJR', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR')))]),
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
                pipeline = make_pipeline(f, XGBClassifier(n_jobs=8, n_estimators=500))
                fitted = pipeline.fit(train, train[LABEL].values)
                predicted = fitted.predict(test)
                folds_results.append((test[LABEL].values, predicted))
                logger.info(
                    classification_report(test[LABEL].values, predicted, labels=labels, digits=2))
                matrix = confusion_matrix(test[LABEL].values, predicted, labels=labels)
                logger.info(matrix)
                conf_dict[caption] += matrix

                result_frame.loc[idx] = [caption, 'Weighted Average', f1_score(test[LABEL].values, predicted, average='weighted'), precision_score(test[LABEL].values, predicted, average='weighted'),
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
