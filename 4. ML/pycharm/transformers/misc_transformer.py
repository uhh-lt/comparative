from sklearn.base import TransformerMixin
import spacy

nlp = spacy.load('en')


class Joiner(TransformerMixin):

    def __init__(self, on=' '):
        self.on = on

    def transform(self, list_of_lists):
        return [self.on.join(l) for l in list_of_lists]

    def fit(self, X, y):
        return self


class Lemmatizer(TransformerMixin):

    def transform(self, documents):
        result = []
        for _doc in documents:
            doc = nlp(_doc)
            result.append([t.lemma_ for t in doc])
        return result


    def fit(self, X, y):
        return self

