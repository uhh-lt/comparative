import re

from pandas import DataFrame
from tools import StopwordRemover, LengthAnalyzer, WordOccurence, BeforeFirstObject
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def build_data_frame(file_name):
    rows = []
    with open(file_name, 'r') as data:
        for line in data:
            text, label = re.split(r'\t', line)
            rows.append({'text': text.strip(), 'label': label.strip()})
    data_frame = DataFrame(rows)
    return data_frame


# nltk.download('stopwords')

DATA = build_data_frame('data/data.txt')


def run_pipeline(model):
    # train, test = train_test_split(DATA, test_size=0.2)
    kf = KFold(n_splits=3, shuffle=True)
    f1 = []
    for train_idx, test_idx in kf.split(DATA['text'].values, DATA['label'].values):
        train = DATA.iloc[train_idx]

        test = DATA.iloc[test_idx]

        union = FeatureUnion(
            [('meta', Pipeline([('length', LengthAnalyzer()), ])),
           #  ('word-occ', Pipeline([('word-occ', WordOccurence(["better", "worse", "?", "than"]))])),

          #   ('bag-of-words-first',
            #  Pipeline([('before-first', BeforeFirstObject(
            #      'OBJECT_A')), ('word-occ', WordOccurence(["better",
            #                                                "worse", "?",
            #                                                "than",
            #                                                "because",
            #                                                "inferior"])
            #                     )]
             #          )),
             #('bag-of-words', Pipeline([('tfidf', CountVectorizer())])),
             ])
        pipeline = Pipeline([
            ('features', union),
            ('model', model)
        ])
        modela = pipeline.fit(train['text'].values, train['label'].values)
        t = union.transform(train['text'].values)
        print(list(t))
        break
        predictions = modela.predict(test['text'].values)
        f1.append(
            f1_score(test['label'].values, predictions, labels=['BETTER', 'WORSE', 'NO_COMP', 'OUT'], average='micro'))


        # print(classification_report(test['label'].values, predictions, labels=['BETTER', 'WORSE', 'NO_COMP','OUT']))
        # print(confusion_matrix(test['label'].values, predictions, labels=['BETTER', 'WORSE', 'NO_COMP','OUT']))
    print(sum(f1) / len(f1))


if __name__ == '__main__':
    run_pipeline(LogisticRegression())
    run_pipeline(MultinomialNB())
    run_pipeline(SVC())
    run_pipeline(LinearSVC())
    run_pipeline(Perceptron())
