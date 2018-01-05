from sklearn.base import BaseEstimator, TransformerMixin
import spacy
class BaseFeature(BaseEstimator, TransformerMixin):

    nlp = spacy.load('en')

    def fit(self,X,y):
        return self
