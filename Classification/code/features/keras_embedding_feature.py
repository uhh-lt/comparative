from .base_feature import BaseFeature
from keras.layers import Dense, Activation
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


class KerasEmbedding(BaseFeature):
    """Boolean feature; checks if the sentences contains the given POS"""

    def __init__(self):
        pass

    def transform(self, sentences):
        extractor = ExtractMiddlePart()


        n_grams = get_all_ngrams(sentences)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(sentences))

        seq = sequence.pad_sequences(tokenizer.texts_to_sequences(list(sentences)), maxlen=10)

        maxlen_ = 10
        embedding_vector_length = 32
        model = Sequential()
        model.add(Embedding(len(n_grams), embedding_vector_length))
        model.add(LSTM(100, return_sequences=True))
        lstm = LSTM(maxlen_)
        model.add(lstm)
        # model.add(Dense(3, activation='softmax'))
        model.add(Activation('linear'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(seq, seq, epochs=1, batch_size=64, verbose=1)
        # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, verbose=1)


        predict = model.predict(seq)
        return predict


class ModelFeature(BaseFeature):

    def __init__(self, model):
        self.model = model

    def transform(self, sentences):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(sentences))
        seq = sequence.pad_sequences(tokenizer.texts_to_sequences(list(sentences)), maxlen=10)
        return self.model.predict(seq)