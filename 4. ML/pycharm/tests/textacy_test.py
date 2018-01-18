import unittest
from textacy.extract import ngrams as t_ngrams
import textacy
import spacy

nlp = spacy.load('en')


class TextacyNgrams(unittest.TestCase):

    def test_filter_punct(self):
        ngrams = sorted(t_ngrams(nlp('Java, Python and Ruby foo, bar'), n=2, filter_punct=True))
        print('Produced bigrams {}'.format(ngrams))
        expected = sorted(['Java Python', 'Python and', 'and Ruby', 'Ruby foo', 'foo bar'])


        self.assertListEqual(sorted([t.text for t in ngrams]), expected)
