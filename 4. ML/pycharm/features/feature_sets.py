from itertools import combinations

from sklearn.pipeline import make_pipeline

from features.contains import ContainsWord, ContainsPos
from features.mean_wordembedding import MeanWordEmbedding
from features.ngram_feature import NGramFeature
from infersent.infersent_feature import InfersentFeature
from transformers.data_extraction import ExtractMiddlePart, ExtractRawSentence
from transformers.n_gram_transformers import NGramTransformer
from util.ngram import get_all_ngrams

CUE_WORDS_WORSE = ["worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior",
                   "mediocre"]

CUE_WORDS_BETTER = ["better", "easier", "faster", "nicer", "wiser", "cooler", "decent", "safer", "superior", "solid",
                    "teriffic"]


def get_feature_names(pipeline):
    lst = pipeline.named_steps['featureunion'].transformer_list
    names = []
    for t in lst:
        names += t[1].steps[-1][1].get_feature_names()
    return names


def get_best_so_far(infersent_model):
    base = [
        (
            'infersent-replace',
            make_pipeline(ExtractMiddlePart(processing='replace'), InfersentFeature(infersent_model))),

        ('contains-w', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_WORSE + CUE_WORDS_BETTER))),

        ('jjr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),
    ]
    return base


def get_all(data, infersent_model):
    unigrams, bigrams, trigrams = _setup_n_grams(data)
    return [
       # ('infersent-m', make_pipeline(ExtractMiddlePart(processing=None), InfersentFeature(infersent_model))),
        ('uni-m', make_pipeline(ExtractMiddlePart(processing=None), NGramTransformer(), NGramFeature(unigrams))),
        ('bi-m', make_pipeline(ExtractMiddlePart(processing=None), NGramTransformer(n=2), NGramFeature(bigrams, n=2))),
        ('tri-m',
         make_pipeline(ExtractRawSentence(processing=None), NGramTransformer(n=3), NGramFeature(trigrams, n=3))),
        ('contains-worse', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_WORSE))),
        ('contains-better', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER))),
        # ('mwe', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding())),
        ('jjr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),
        ('jjs-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJS'))),
        ('jj-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJ'))),
        ('md-m', make_pipeline(ExtractMiddlePart(), ContainsPos('MD'))),
        ('rbr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('RBR'))),
        ('rbs-m', make_pipeline(ExtractMiddlePart(), ContainsPos('RBS'))),
        ('ls-m', make_pipeline(ExtractMiddlePart(), ContainsPos('LS'))),
        ('cd-m', make_pipeline(ExtractMiddlePart(), ContainsPos('CD'))),
        ('fw-m', make_pipeline(ExtractMiddlePart(), ContainsPos('FW'))),
        ('rbr', make_pipeline(ExtractMiddlePart(), ContainsPos('RBR'))),
        ('rbs', make_pipeline(ExtractMiddlePart(), ContainsPos('RBS'))),
    ]


def feature_grid(data, infersent_model):
    unigrams, bigrams, trigrams = _setup_n_grams(data)

    feat = [
        #   ('infersent-m', make_pipeline(ExtractMiddlePart(processing=None), InfersentFeature(infersent_model))),
        # ('uni-m', make_pipeline(ExtractMiddlePart(processing=None), NGramTransformer(), NGramFeature(unigrams))),
        # ('bi-m', make_pipeline(ExtractMiddlePart(processing=None), NGramTransformer(n=2), NGramFeature(bigrams))),
        ('tri-m', make_pipeline(ExtractRawSentence(processing=None), NGramTransformer(n=3), NGramFeature(trigrams))),
        ('contains-worse', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_WORSE))),
        ('contains-better', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER))),
        ('mwe', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding())),
        ('jjr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),
        ('jjs-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJS'))),
        ('jj-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJ'))),
        ('md-m', make_pipeline(ExtractMiddlePart(), ContainsPos('MD'))),
        ('rbr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('RBR'))),
        ('rbs-m', make_pipeline(ExtractMiddlePart(), ContainsPos('RBS'))),
        ('ls-m', make_pipeline(ExtractMiddlePart(), ContainsPos('LS'))),
        ('cd-m', make_pipeline(ExtractMiddlePart(), ContainsPos('CD'))),
        ('fw-m', make_pipeline(ExtractMiddlePart(), ContainsPos('FW'))),
    ]
    feat_ = list(combinations(feat, 1)) + list(combinations(feat, len(feat)))
    print('Combinations to test: {}'.format(len(feat_)))
    return feat_


def _setup_n_grams(d):
    return get_all_ngrams(d['raw_text'].values, n=1), get_all_ngrams(d['raw_text'].values, n=2), get_all_ngrams(
        d['raw_text'].values, n=3)
