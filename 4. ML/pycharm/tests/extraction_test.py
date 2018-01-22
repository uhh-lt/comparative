import unittest
import pandas as pd
from bs4 import BeautifulSoup
import transformers.data_extraction as tr

frame = pd.DataFrame.from_csv(path='../data/train-data.csv')
frame['raw_text'] = frame.apply(
    lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
    axis=1)

sentence = "It's definitely faster than using MySQL or PostgreSQL for what you want to do though... -  Aaron Aug 18 '11 at 18:56"


class ExtractionTest(unittest.TestCase):

    def test_extract_middle_part_with_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            ex = tr.ExtractMiddlePart()
            res = ex.transform(frame)[counter]

            counter += 1
            self.assertTrue(len(text) > len(res))
            self.assertTrue(a in res)
            self.assertTrue(b in res)

    def test_extract_middle_part_without_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            ex = tr.ExtractMiddlePart(processing='remove')
            res = ex.transform(frame)[counter]

            counter += 1
            self.assertTrue(len(text) > len(res))
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)

    def test_extract_middle_part_with_objects_replaced(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            ex = tr.ExtractMiddlePart(processing='replace')
            res = ex.transform(frame)[counter]

            counter += 1

            self.assertTrue(res.count('OBJECT') is 2)

    def test_extract_middle_part_with_objects_replaced_dist(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            ex = tr.ExtractMiddlePart(processing='replace_dist')
            res = ex.transform(frame)[counter]

            counter += 1

            self.assertTrue(res.count('OBJECT_A') is 1)
            self.assertTrue(res.count('OBJECT_B') is 1)
            self.assertTrue(res.index('OBJECT_A') < res.index('OBJECT_B'))

    def test_extract_full_sentence_with_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            ex = tr.ExtractRawSentence()
            res = ex.transform(frame)[counter]
            counter += 1
            self.assertEqual(text, res)

    def test_extract_full_sentence_without_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            ex = tr.ExtractRawSentence(processing='remove')
            res = ex.transform(frame)[counter]
            counter += 1
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)

    def test_extract_full_sentence_replace_objects_uni(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            ex = tr.ExtractRawSentence(processing='replace')
            res = ex.transform(frame)[counter]
            counter += 1
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)
            self.assertTrue(res.count('OBJECT') is 2)

    def test_extract_full_sentence_replace_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']

            ex = tr.ExtractRawSentence(processing='replace_dist')
            res = ex.transform(frame)[counter]
            counter += 1
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)
            self.assertTrue(res.count('OBJECT_A') is 1)
            self.assertTrue(res.count('OBJECT_B') is 1)

    def test_extract_first_with_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']

            ex = tr.ExtractFirstPart()
            res = ex.transform(frame)[counter]
            counter += 1
            self.assertTrue((a in res) ^ (b in res))

    def test_extract_first_without_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']

            ex = tr.ExtractFirstPart(processing='remove')
            res = ex.transform(frame)[counter]
            counter += 1
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)

    def test_extract_first_with_replace(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            print(text)
            ex = tr.ExtractFirstPart(processing='replace')
            res = ex.transform(frame)[counter]
            print(res)
            counter += 1
            self.assertTrue(res.count('OBJECT') is 1)
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)

    def test_extract_first_with_replace_dist(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            print(text)
            ex = tr.ExtractFirstPart(processing='replace_dist')
            res = ex.transform(frame)[counter]
            print(res)
            counter += 1
            self.assertTrue(res.count('OBJECT_A') is 1)
            self.assertTrue('OBJECT_B' not in res)
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)

    def test_extract_last_with_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']

            ex = tr.ExtractLastPart()
            res = ex.transform(frame)[counter]
            print(text)
            print(res)
            counter += 1
            self.assertTrue((a in res) ^ (b in res))

    def test_extract_last_without_objects(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']

            ex = tr.ExtractLastPart(processing='remove')
            res = ex.transform(frame)[counter]
            counter += 1
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)

    def test_extract_last_with_replace(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            print(text)
            ex = tr.ExtractLastPart(processing='replace')
            res = ex.transform(frame)[counter]
            print(res)
            counter += 1
            self.assertTrue(res.count('OBJECT') is 1)
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)

    def test_extract_last_with_replace_dist(self):
        counter = 0
        for index, row in frame.iterrows():
            text, a, b = row['raw_text'], row['a'], row['b']
            print(text)
            ex = tr.ExtractLastPart(processing='replace_dist')
            res = ex.transform(frame)[counter]
            print(res)
            counter += 1
            self.assertTrue(res.count('OBJECT_A') is 1)
            self.assertTrue('OBJECT_B' not in res)
            self.assertTrue(a not in res)
            self.assertTrue(b not in res)
