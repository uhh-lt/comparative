import itertools

from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from data.data import load_data, k_folds
from features.feature_sets import _setup_n_grams
from features.keras_embedding_feature import KerasEmbedding, ModelFeature
from infersent.infersent_feature import *
from transformers.data_extraction import *
from util.feature_builder import *
from keras.layers import Dense, Activation, Conv1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from data.data import load_data
from transformers.data_extraction import ExtractMiddlePart
from util.ngram import get_all_ngrams


def pad(texts, maxlen):
    n_grams = get_all_ngrams(texts)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(texts))

    return sequence.pad_sequences(tokenizer.texts_to_sequences(list(texts)), maxlen=maxlen), len(n_grams)


def perform_classification(classifier, data, labels, min_f1=0.66):
    f1 = 0
    acc = 0
    for train, test in k_folds(3, data, random_state=222):
        raw_text = train['raw_text'].values
        # extractor = ExtractMiddlePart()
        #
        # input, in_voc = pad(raw_text, 100)
        # out_len = 5
        # output, out_voc = pad(extractor.transform(train), out_len)
        #
        # embedding_vector_length = 300
        # model = Sequential()
        # model.add(Embedding(in_voc, embedding_vector_length))
        # # model.add(LSTM(64, return_sequences=True))
        # model.add(LSTM(100, return_sequences=True))
        # lstm = LSTM(out_len)
        # model.add(lstm)
        # # model.add(Dense(3, activation='softmax'))
        # model.add(Activation('linear'))
        #
        # model.compile(loss='categorical_crossentropy', optimizer='adam')
        # model.fit(input, output, epochs=1, batch_size=256, verbose=1)

        infersent_model = initialize_infersent(raw_text)
        pipeline = make_pipeline(FeatureUnion(
            [
                #
                # ('jjr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),
                #  ('jjs', make_pipeline(ExtractMiddlePart(), ContainsPos('JJS')))]),
                # ('wtf', make_pipeline(ExtractMiddlePart(), ModelFeature(model))),
                ('infersent-m',
                 make_pipeline(ExtractMiddlePart(processing='replace_dist'), InfersentFeature(infersent_model)))]),
            # ('infersent-f', make_pipeline(ExtractRawSentence(processing=None), InfersentFeature(infersent_model)))]),
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
    _data = load_data('train-data.csv', min_confidence=0.67, binary=False, source=source)
    _data_other_merged = _data.copy()
    _data_other_merged['label'] = _data_other_merged.apply(
        lambda row: row['label'] if row['label'] != 'OTHER' else 'NONE', axis=1)

    _data_sets = [
        # ('binary', ['ARG', 'NONE'], load_data('train-data.csv', min_confidence=0, binary=True, source=source))
        ('3 Label, OTHER merged', ['BETTER', 'WORSE', 'NONE'], _data_other_merged),
        # ('Binary', ['ARG', 'NONE'], load_data('train-data.csv', binary=True, source=source))
    ]

    # Ridge 0.76
    classifiers = [XGBClassifier()]
    # unigrams, bigrams, trigrams = _setup_n_grams(_data)

    # features = feature_grid(_data, infersent_model)
    for headline, label, data in _data_sets:
        print('# {}'.format(headline))
        for classifier in classifiers:
            print('## {}'.format(type(classifier)))

            f1 = perform_classification(classifier, data, label)

        print('\n\n')
    print('\n\n')


if __name__ == '__main__':
    # experiment_b()
    experiment_a()
