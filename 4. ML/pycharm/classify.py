from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from features.ngram_feature import NGramFeature
from transformers.data_extraction import *
from transformers.n_gram_transformers import NGramTransformer
from util.feature_builder import *
from util.ngram import get_all_ngrams

CUE_WORDS_WORSE = ["worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior",
                   "mediocre"]

CUE_WORDS_BETTER = ["better", "easier", "faster", "nicer", "wiser", "cooler", "decent", "safer", "superior", "solid",
                    "teriffic"]


def load_data(file_name, min_confidence=1, binary=False):
    print('### Minimum Confidence {}'.format(min_confidence))
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    frame['raw_text'] = frame.apply(
        lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
        axis=1)
    if binary:
        frame['label'] = frame.apply(lambda row: row['label'] if row['label'] == 'NONE' else 'ARG', axis=1)
    return shuffle(frame)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits, random_state=42)
    for train_index, test_index in k_fold.split(data,
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def setup_n_grams(d):
    return get_all_ngrams(d['raw_text'].values, n=1), get_all_ngrams(d['raw_text'].values, n=2)


def perform_classification(pipeline, data):
    f1 = 0
    acc = 0
    for train, test in split_data(3, data):
        fitted = pipeline.fit(train, train['label'])
        predicted = fitted.predict(test)
        f1 += f1_score(test['label'].values, predicted, average='weighted', labels=_labels)
        acc += accuracy_score(test['label'].values, predicted)

        print(classification_report(test['label'].values, predicted, labels=_labels))
        print(confusion_matrix(test['label'].values, predicted, labels=_labels))

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


def linear_svc(features, data, grid=False):
    print('********* SVC *********')

    pipeline = make_pipeline(FeatureUnion(features), LinearSVC())

    param_grid = {
        'featureunion__ngram__extractrawsentence__processing': ['remove', 'replace', 'replace_dist'],
        'featureunion__bi-m__extractmiddlepart__processing': ['remove', 'replace', 'replace_dist'],
        'featureunion__uni-m__extractmiddlepart__processing': ['remove', 'replace', 'replace_dist'],
        'featureunion__ngram__ngramfeature__base_n_grams': [unigrams],
        'featureunion__bi-m__ngramfeature__base_n_grams': [bigrams],
        'featureunion__uni-m__ngramfeature__base_n_grams': [unigrams]
    }

    if grid:
        perform_grid_search(pipeline, data, param_grid)
    else:
        return perform_classification(pipeline, data)


def logistic_regression(features, data, grid=False):
    print('********* LogisticRegression *********')
    pipeline = make_pipeline(FeatureUnion(features), LogisticRegression())
    if grid:
        perform_grid_search(pipeline, data, {})
    else:
        return perform_classification(pipeline, data)


def sgd(features, data, grid=False):
    print('********* SGD *********')
    pipeline = make_pipeline(FeatureUnion(features), SGDClassifier())
    if grid:
        perform_grid_search(pipeline, data, {})
    else:
        return perform_classification(pipeline, data)


if __name__ == '__main__':
    _labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
    # _labels = ['ARG', 'NONE']
    #  _data = load_data('train-data.csv', binary=False)
    _data = load_data('train-data-with-sent.csv', binary=False)
    unigrams, bigrams = setup_n_grams(_data)
    print('Build n-grams')
    _processing = 'replace'
    best_so_far = [

        ('better-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER))),
        ('bw-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),
        ('ngram',
         make_pipeline(ExtractRawSentence(processing=_processing), NGramTransformer(n=1), NGramFeature(unigrams))),
        (
            'bi-m',
            make_pipeline(ExtractMiddlePart(processing=_processing), NGramTransformer(n=2), NGramFeature(bigrams))),
        (
            'uni-m',
            make_pipeline(ExtractMiddlePart(processing=_processing), NGramTransformer(n=1), NGramFeature(unigrams))),

    ]
    f1 = 0
    print('Build features')
    # f1_a = sgd(best_so_far, _data, grid=False)
    # f1_b = logistic_regression(best_so_far, _data, grid=False)
    # f1_c = linear_svc(best_so_far, _data, grid=False)
    cf = [LinearSVC(loss='hinge'),RandomForestClassifier(), ExtraTreesClassifier(), RidgeClassifier(), GradientBoostingClassifier(), SVC(),
          MultinomialNB(), GaussianNB(), LinearSVC(), LogisticRegression(), SGDClassifier()]
    for c in cf:
        try:
            print('********* {} *********'.format(type(c)))
            pipeline = make_pipeline(FeatureUnion(best_so_far), c)
            f1 += perform_classification(pipeline, _data)
        except Exception as e:
            print(e)

    print('==============\nAverage of all averages F1 {}'.format((f1) / len(cf)))
