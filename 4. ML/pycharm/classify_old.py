import itertools

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from infersent.infersent_feature import *
from transformers.data_extraction import ExtractMiddlePart
from util.data_utils import load_data, k_folds


def perform_classification(classifier, data, labels, min_f1=0.66):
    f1 = 0
    acc = 0
    for train, test in k_folds(3, data, random_state=222):
        raw_text = train['raw_text'].values
        infersent_model = initialize_infersent(raw_text)
        pipeline = make_pipeline(FeatureUnion(
            [
                ('infersent-m',
                 make_pipeline(ExtractMiddlePart(processing='replace_dist'), InfersentFeature(infersent_model)))]),
            classifier)

        fitted = pipeline.fit(train, train['label'])
        predicted = fitted.predict(test)
        f1 += f1_score(test['label'].values, predicted, average='weighted', labels=labels)
        acc += accuracy_score(test['label'].values, predicted)

        # get_misclassified(predicted, test)

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
    _data = load_data('data.csv', min_confidence=0, binary=False, source=source)

    # Ridge 0.76
    classifiers = [XGBClassifier()]
    for classifier in classifiers:
        print('## {}'.format(type(classifier)))

        f1 = perform_classification(classifier, _data, ['BETTER', 'WORSE', 'NONE'])

    print('\n\n')


if __name__ == '__main__':
    # experiment_b()
    experiment_a()
