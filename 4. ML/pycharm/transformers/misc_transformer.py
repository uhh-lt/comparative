from sklearn.base import TransformerMixin


class Joiner(TransformerMixin):

    def __init__(self, on=' '):
        self.on = on

    def transform(self, list_of_lists):
        return [self.on.join(l) for l in list_of_lists]

    def fit(self, X, y):
        return self
