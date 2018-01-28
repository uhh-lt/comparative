import itertools

from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from data.data_loader import *
from features.feature_sets import *
from infersent.infersent_feature import *
from transformers.data_extraction import *
from util.feature_builder import *


def perform_classification(pipeline, data, labels, silent=False):
    f1 = 0
    acc = 0
    for train, test in k_folds(3, data):
        fitted = pipeline.fit(train, train['label'])
        predicted = fitted.predict(test)
        f1 += f1_score(test['label'].values, predicted, average='weighted', labels=labels)
        acc += accuracy_score(test['label'].values, predicted)
        if not silent:
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
    _data_other_merged = _data.copy()
    _data_other_merged['label'] = _data_other_merged.apply(
        lambda row: row['label'] if row['label'] != 'OTHER' else 'NONE', axis=1)

    _data_sets = [('4 Label', ['BETTER', 'WORSE', 'OTHER', 'NONE'], _data),
                  ('3 Label, w/o OTHER', ['BETTER', 'WORSE', 'NONE'], _data[_data['label'] != 'OTHER']),
                  #      ('3 Label, OTHER merged', ['BETTER', 'WORSE', 'NONE'], _data_other_merged),
                  ('Binary', ['ARG', 'NONE'], load_data('train-data.csv', binary=True, source=source))
                  ]

    res = []
    # Ridge 0.76
    classifiers = [RidgeClassifier()]

    infersent_model = initialize_infersent(_data['raw_text'].values)
    base = get_best_so_far(infersent_model)

    features = feature_grid(_data, infersent_model)
    for feature_set in features:
        print(feature_set)
        for headline, label, data in _data_sets:
            print('# {}'.format(headline))
            for classifier in classifiers:
                try:
                    print('## {}'.format(type(classifier)))
                    pipeline = make_pipeline(FeatureUnion(base), classifier)
                    f1 = perform_classification(pipeline, data, label, silent=False)
                    res.append((f1, classifier))
                except Exception as e:
                    print(e)
            print('\n\n')
        print('\n\n')
    for t in sorted(res, key=lambda k: k[0]):
        print(t)


if __name__ == '__main__':
    # experiment_b()
    experiment_a()
