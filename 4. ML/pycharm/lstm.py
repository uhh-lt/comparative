from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy
from data.data import load_data
from transformers.data_extraction import ExtractMiddlePart
from util.ngram import get_all_ngrams

extractor = ExtractMiddlePart()
_data = load_data('train-data.csv', min_confidence=0, binary=False)
_data['label'] = _data.apply(
    lambda row: row['label'] if row['label'] != 'OTHER' else 'NONE', axis=1)

middle = extractor.transform(_data)
n_grams = get_all_ngrams(middle)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(middle))

maxlen_ = maxlen = 50
seq = sequence.pad_sequences(tokenizer.texts_to_sequences(list(middle)), maxlen_)
labels = LabelBinarizer().fit_transform(_data['label'].values)

X_train, X_test, y_train, y_test = train_test_split(seq, labels)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(len(n_grams), embedding_vector_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(1000, return_sequences=True))
lstm = LSTM(maxlen_)
model.add(lstm)
# model.add(Dense(3, activation='softmax'))
model.add(Activation('linear'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, X_train, epochs=1, batch_size=64, verbose=1)
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, verbose=1)


foo = sequence.pad_sequences(
    tokenizer.texts_to_sequences(["Ruby is better than Python", "Java is better than Scala", "foo bar baz"]),
    maxlen_)
predict = model.predict(foo)
print(predict)
print(numpy.linalg.norm(foo[0] - foo[1]))
print(numpy.linalg.norm(foo[0] - foo[2]))
print(numpy.linalg.norm(foo[1] - foo[2]))
