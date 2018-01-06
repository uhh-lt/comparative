from sklearn.base import TransformerMixin

class SentenceSplit(TransformerMixin):

    def transform(self, documents):

        return documents

    def fit(self,X,y):
        print(X,y)
        return self
