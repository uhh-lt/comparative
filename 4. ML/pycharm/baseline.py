from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

from features.ngram_feature import NGramFeature
from transformers.data_extraction import ExtractRawSentence
from transformers.n_gram_transformers import NGramTransformer
from util.data_utils import load_data, k_folds
from util.misc_utils import latex_table, latex_table_bin
from util.ngram_utils import get_all_ngrams

import datetime



now = datetime.datetime.now()
file_name = 'baseline-{}-{}-{}:{}'.format(now.day, now.month, now.hour, now.minute)


def create_baseline(data, labels, table_fkt):
    classifiers = [DummyClassifier(random_state=42), DummyClassifier(strategy='most_frequent')]

    for classifier in classifiers:
        print('=========== {} =========== '.format(classifier))

        res = []

        for train, test in k_folds(5, data):
            ngram_base = ExtractRawSentence().transform(train)
            unigrams = get_all_ngrams(ngram_base, n=1)
            pipeline = make_pipeline(ExtractRawSentence(), NGramTransformer(n=1), NGramFeature(unigrams), classifier)
            fitted = pipeline.fit(train, train['label'].values)
            predicted = fitted.predict(test)
            print(classification_report(test['label'].values, predicted, labels=labels))
            print(f1_score(test['label'].values, predicted, average='weighted',
                           labels=labels))

            res.append((f1_score(test['label'].values, predicted, average='weighted',
                                 labels=labels), (test['label'].values, predicted)))

            print("===========")
        res = sorted(res, key=lambda x: x[0])
        table_fkt([res[0][1]] + [res[2][1]] + [res[4][1]], 'cap')


print('# BINARY CLASSES BASELINE')
_data_bin = load_data('data.csv', min_confidence=0, binary=True)
create_baseline(_data_bin, ['ARG', 'NONE'], latex_table_bin)

print('# THREE CLASSES BASELINE')
_data = load_data('data.csv', min_confidence=0, binary=False)
create_baseline(_data, ['BETTER', 'WORSE', 'NONE'], latex_table)
