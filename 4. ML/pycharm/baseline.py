from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, \
    precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from collections import OrderedDict

from features.ngram_feature import NGramFeature
from transformers.data_extraction import ExtractRawSentence
from transformers.n_gram_transformers import NGramTransformer
from util.data_utils import load_data, k_folds
from util.ngram_utils import get_all_ngrams
from pprint import pprint


_data = load_data('data.csv', min_confidence=0, binary=False)
_data['label'] = _data.apply(
    lambda row: row['label'] if row['label'] != 'OTHER' else 'NONE', axis=1)
# RidgeClassifier(), DummyClassifier(random_state=42),
classifiers = [RidgeClassifier(), DummyClassifier(strategy='most_frequent')]


def latex_table(res, cap):
    b_row = '\\texttt{BETTER}\t & '
    w_row = '\\texttt{WORSE}\t & '
    n_row = '\\texttt{NONE}\t & '
    a_row = 'average\t & '
    for v in res:
        precision, recall, f1, support = precision_recall_fscore_support(v[0], v[1], average=None,
                                                                         labels=['BETTER', 'WORSE', 'NONE'])
        b_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[0], recall[0], f1[0])
        w_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[1], recall[1], f1[1])
        n_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision[2], recall[2], f1[2])
        precision, recall, f1, support = precision_recall_fscore_support(v[0], v[1], average='weighted',
                                                                         labels=['BETTER', 'WORSE', 'NONE'])
        a_row += '{:04.2f}\t & {:04.2f}\t & {:04.2f}\t &'.format(precision, recall, f1)

    b_row = b_row[:-1] + '\\\ '
    w_row = w_row[:-1] + '\\\ '
    n_row = n_row[:-1] + '\\\ \midrule '
    a_row = a_row[:-1] + '\\\ \\bottomrule'
    print("""
    \\begin{table}[h]
\centering""")
    print('\caption{{ {} }}'.format(cap))
    print('\label{{tbl:{} }}'.format(cap.lower().replace(' ', '_')))

    print("""\\begin{tabular}{@{}lccccccccc@{}}
\\toprule
      & \multicolumn{3}{c}{Worst} & \multicolumn{3}{c}{Average} & \multicolumn{3}{c}{Best}  \\\ \midrule
                 & Precision  & Recall & F1   & Precision  & Recall  & F1    & Precision & Recall & F1   \\\ \\toprule

    """)
    print(b_row)
    print(w_row)
    print(n_row)
    print(a_row)
    print("""
    \end{tabular}
\end{table}

    """)


for classifier in classifiers:
    print(classifier)

    res = []

    for train, test in k_folds(5, _data):
        ngram_base = []  # ExtractRawSentence().transform(train)
        unigrams = []  # get_all_ngrams(ngram_base, n=1)
        # pipeline = make_pipeline(ExtractRawSentence(), NGramTransformer(n=1), NGramFeature(unigrams), classifier)
        pipeline = make_pipeline(ExtractRawSentence(), CountVectorizer(), classifier)
        fitted = pipeline.fit(train, train['label'].values)
        predicted = fitted.predict(test)
        print(classification_report(test['label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE']))
        print(f1_score(test['label'].values, predicted, average='weighted',
                       labels=['BETTER', 'WORSE', 'NONE']))

        res.append((f1_score(test['label'].values, predicted, average='weighted',
                            labels=['BETTER', 'WORSE', 'NONE']), (test['label'].values, predicted)))



        #latex_table(d, 'Majority Class Baseline')
        # print(confusion_matrix(test['label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE']))
        print("===========")
    res = sorted(res, key=lambda x: x[0])
    latex_table([res[0][1]]+[res[2][1]]+[res[4][1]], 'cap')
