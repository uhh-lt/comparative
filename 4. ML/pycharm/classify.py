import itertools
from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import Binarizer, OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from features.ngram_feature import NGramFeature
from transformers.data_extraction import *
from transformers.n_gram_transformers import NGramTransformer
from util.feature_builder import *
from infersent.infersent_feature import *
from util.ngram import get_all_ngrams

CUE_WORDS_WORSE = ["worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior",
                   "mediocre"]

CUE_WORDS_BETTER = ["better", "easier", "faster", "nicer", "wiser", "cooler", "decent", "safer", "superior", "solid",
                    "teriffic"]


def load_data(file_name, min_confidence=0.67, binary=False, source=None):
    print('### Minimum Confidence {}'.format(min_confidence))
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    frame['raw_text'] = frame.apply(
        lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
        axis=1)
    if binary:
        frame['label'] = frame.apply(lambda row: row['label'] if row['label'] == 'NONE' else 'ARG', axis=1)
    if source is not None:
        frame = frame[frame['type'] == source]
    return shuffle(frame)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits, random_state=1333)
    for train_index, test_index in k_fold.split(data,
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def setup_n_grams(d):
    return get_all_ngrams(d['raw_text'].values, n=1), get_all_ngrams(d['raw_text'].values, n=2), get_all_ngrams(
        d['raw_text'].values, n=3)


def perform_classification(pipeline, data, labels):
    f1 = 0
    acc = 0
    for train, test in split_data(3, data):
        fitted = pipeline.fit(train, train['label'])
        predicted = fitted.predict(test)
        f1 += f1_score(test['label'].values, predicted, average='weighted', labels=labels)
        acc += accuracy_score(test['label'].values, predicted)

        print(classification_report(test['label'].values, predicted, labels=labels))
        print(confusion_matrix(test['label'].values, predicted, labels=labels))

    print('Average F1 {} | Accuracy {}'.format((f1 / 3), (acc / 3)))
    return f1 / 3


def perform_grid_search(pipeline, data, param_grid):
    print(pipeline.get_params().keys())

    cv = GridSearchCV(pipeline, param_grid=param_grid, cv=StratifiedKFold(n_splits=3, random_state=42),
                      scoring="f1_weighted", verbose=10)
    cv.fit(data, data['label'])

    print("Best parameters set found on development set:")
    print(cv.best_params_)
    print(cv.best_score_)


def get_feature_names(pipeline):
    lst = pipeline.named_steps['featureunion'].transformer_list
    names = []
    for t in lst:
        names += t[1].steps[-1][1].get_feature_names()
    return names


def experiment_b():
    types = set(['brands', 'compsci', 'jbt'])
    train_types = set(itertools.combinations(types, 2))

    for train_type in train_types:
        a, b = train_type
        test_type = list(types - set(train_type))
        print('*** Train on {} {} Test on {}'.format(a, b, test_type))
        _train = load_data('train-data.csv', source=a).append(load_data('train-data.csv', source=b))
        _test = load_data('train-data.csv', source=test_type[0])
        print(len(_train))
        _dict = _train.append(_test)
        model = initialize_infersent(_dict['raw_text'].values)

        pipe = make_pipeline(
            FeatureUnion([('infersent-m', make_pipeline(ExtractMiddlePart(processing=None), InfersentFeature(model)))]),
            LinearSVC())
        print(pipe)
        labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
        fitted = pipe.fit(_train, _train['label'].values)
        predicted = fitted.predict(_test)

        print(classification_report(_test['label'].values, predicted, labels=labels))
        print(confusion_matrix(_test['label'].values, predicted, labels=labels))


def experiment_a(source=None):
    global features
    _data = load_data('train-data.csv', binary=False, source=source)
    # _data_other_merged = _data.copy()
    # _data_other_merged['label'] = _data_other_merged.apply(   lambda row: row['label'] if row['label'] != 'OTHER' else 'NONE', axis=1)
    # unigrams, bigrams, trigrams = setup_n_grams(_data)
    _data_sets = [('4 Label', ['BETTER', 'WORSE', 'OTHER', 'NONE'], _data),
                  # ('3 Label, w/o OTHER', ['BETTER', 'WORSE', 'NONE'], _data[_data['label'] != 'OTHER']),
                  # ('3 Label, OTHER merged', ['BETTER', 'WORSE', 'NONE'], _data_other_merged),
                  # ('Binary', ['ARG', 'NONE'], load_data('train-data.csv', binary=True,source=source))
                  ]
    print('Build n-grams')
    model = initialize_infersent(_data['raw_text'].values)
    best_so_far = [[
        ('infersent-m', make_pipeline(ExtractMiddlePart(processing=None), InfersentFeature(model))),

    ]]
    f1 = 0
    print('Build features')
    classifiers = [LinearSVC()]
    features = best_so_far  # list(itertools.combinations(candidates, 1))
    for feature_set in features:
        print(feature_set)
        for headline, label, data in _data_sets:
            print('*** {}'.format(headline))
            for classifier in classifiers:
                try:
                    print('********* {} *********'.format(type(classifier)))
                    pipeline = make_pipeline(FeatureUnion(feature_set), classifier)
                    f1 += perform_classification(pipeline, data, label)
                except Exception as e:
                    print(e)


if __name__ == '__main__':
    # experiment_b()
    experiment_a()
