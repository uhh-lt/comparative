from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from features.ngram import *
from features.contains import *

def build_contains_word(words, extractor,as_union=False):
    """builds a boolean feature for each word in the part extracted by the extractor"""
    features = []
    for word in words:
        features.append(('contains-' + str(type(extractor())) + word, make_pipeline(extractor(), ContainsWord(word))))
    if as_union:
        return [FeatureUnion(features)]
    return features