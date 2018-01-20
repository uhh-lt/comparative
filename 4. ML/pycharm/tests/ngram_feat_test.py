import unittest

import spacy
from sklearn.pipeline import make_pipeline

import util.ngram as ngram_helper
from features import ngram_feature
from transformers.n_gram_transformers import NGramTransformer

nlp = spacy.load('en')

documents = ['Python is better than Ruby',
             'Java is worse than Scala',
             'Java, Python and Ruby foo, bar'
             ]


class NGramTest(unittest.TestCase):

    def test_unigram_of_documents(self):
        ngrams = sorted(ngram_helper.get_all_ngrams(documents, 1, 1, filter_punct=True))
        expected_ngrams = sorted(['Python', 'is', 'better', 'than', 'and', 'Ruby',
                                  'Java', 'worse', 'Scala', 'foo', 'bar'])
        self.assertListEqual(ngrams, expected_ngrams)

    def test_ngram_transformer(self):
        test_sentence = 'Haskell is neither better nor worse than Python and Ruby'
        transformer = NGramTransformer()
        res = transformer.transform([test_sentence])
        print(res)
        self.assertListEqual(sorted(res[0]), sorted(['Haskell', 'is', 'neither', 'better', 'nor', 'worse', 'than',
                                                     'Python', 'and', 'Ruby']))

    def test_unigram(self):
        ngrams = sorted(ngram_helper.get_all_ngrams(documents, 1, 1, filter_punct=True))

        test_sentence = 'Haskell is neither better nor worse than Python and Ruby'

        transformer = NGramTransformer()
        transformed = transformer.transform([test_sentence])

        feat_with_punct = ngram_feature.NGramFeature(ngrams)
        result = feat_with_punct.transform(transformed)
        print(result)
        self.assertListEqual([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1], result.tolist()[0])

    @unittest.skip
    def test_bigram_of_documents(self):
        bigrams = sorted(ngram_helper.get_all_ngrams(documents, 2, 1, filter_punct=True))
        expected = sorted(['Python is', 'is better', 'better than', 'than Ruby',
                           'Java is', 'is worse', 'worse than', 'than Scala',
                           'Java Python', 'Python and', 'and Ruby', 'Ruby foo'])
        # 'foo bar' & 'Java is' ??
        self.assertListEqual(bigrams, expected)

    def test_ngram_transformer_bigram(self):
        test_sentence = 'Haskell is neither better nor worse than Python and Ruby'
        transformer = NGramTransformer(n=2)
        res = transformer.transform([test_sentence])
        self.assertListEqual(sorted(res[0]),
                             sorted(['Haskell is', 'is neither', 'neither better',
                                     'better nor', 'nor worse', 'worse than', 'than Python',
                                     'Python and', 'and Ruby']))

    def test_bigram(self):
        ngrams = sorted(ngram_helper.get_all_ngrams(documents, 2, 1, filter_punct=True))
        print(ngrams)
        test_sentence = 'Haskell is neither better nor worse than Python and Ruby'

        transformer = NGramTransformer(n=2)
        transformed = transformer.transform([test_sentence])

        feat_with_punct = ngram_feature.NGramFeature(ngrams)
        result = feat_with_punct.transform(transformed)
        print(result)
        self.assertListEqual([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], result.tolist()[0])

    def test_pipeline(self):
        ngrams = sorted(ngram_helper.get_all_ngrams(documents, 1, 1, filter_punct=True))
        test_sentence = 'Haskell is neither better nor worse than Python and Ruby'

        p = make_pipeline(NGramTransformer(), ngram_feature.NGramFeature(ngrams))
        p.fit(test_sentence, True)



if __name__ == '__main__':
    unittest.main()
