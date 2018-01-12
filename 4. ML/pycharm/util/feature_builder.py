from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from features.ngram import *
from features.contains import *

def build_contains_word(words, extractor):
    """builds a boolean feature for each word in the part extracted by the extractor"""
    features = []
    for word in words:
        features.append(('contains-' + word, make_pipeline(extractor(), ContainsWord(word))))
    return features