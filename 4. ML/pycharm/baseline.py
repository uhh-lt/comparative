from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from data.data import load_data, k_folds
from features.feature_sets import _setup_n_grams
from features.ngram_feature import NGramFeature
from transformers.data_extraction import ExtractMiddlePart, ExtractRawSentence
from transformers.n_gram_transformers import NGramTransformer
from util.ngram import get_all_ngrams

_data = load_data('train-data.csv', min_confidence=0, binary=False)
_data['label'] = _data.apply(
    lambda row: row['label'] if row['label'] != 'OTHER' else 'NONE', axis=1)

classifiers = [LogisticRegression(), DummyClassifier(random_state=42), DummyClassifier(strategy='most_frequent')]

for classifier in classifiers:
    print(classifier)
    for train, test in k_folds(3, _data):
        ngram_base = ExtractRawSentence().transform(train)
        unigrams = get_all_ngrams(ngram_base, n=1)
        pipeline = make_pipeline(ExtractRawSentence(), NGramTransformer(), NGramFeature(unigrams), classifier)
        fitted = pipeline.fit(train, train['label'].values)
        predicted = fitted.predict(test)
        print(classification_report(test['label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE']))
        print(confusion_matrix(test['label'].values, predicted, labels=['BETTER', 'WORSE', 'NONE']))
    print("===========")
