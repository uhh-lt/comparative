from sklearn.base import TransformerMixin
import spacy
import re

from util.data_utils import CUE_WORDS_WORSE, CUE_WORDS_BETTER

nlp = spacy.load('en_core_web_lg')


class Joiner(TransformerMixin):

    def __init__(self, on=' '):
        self.on = on

    def transform(self, list_of_lists):
        return [self.on.join(l) for l in list_of_lists]

    def fit(self, X, y):
        return self

class ReplaceCueWord(TransformerMixin):

    def __init__(self, cue_words, replacement):
        self.cue_words = cue_words
        self.replacement = replacement

    def transform(self, documents):
        result = []

        reg = {}
        for cue in self.cue_words:
            reg[cue] = re.compile(re.escape(cue), re.IGNORECASE)

        for doc in documents:
            repl = doc
            for key, rx in reg.items():
                repl = rx.sub(self.replacement, repl)
            result.append(repl)
        return result

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

