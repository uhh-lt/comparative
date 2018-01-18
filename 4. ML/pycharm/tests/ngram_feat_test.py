import unittest

import spacy

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
        print(ngrams)

        test_sentence = 'Haskell is neither better nor worse than Python and Ruby'

        transformer = NGramTransformer()
        transformed = transformer.transform([test_sentence])

        feat_with_punct = ngram_feature.NGramFeature(ngrams)
        result = feat_with_punct.transform(transformed)
        print(result)
        self.assertListEqual([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1], result.tolist()[0])


if __name__ == '__main__':
    unittest.main()
