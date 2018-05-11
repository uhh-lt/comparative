"""Adds the prepared sentence for the path embeddings and the sentence embeddings (InferSent) to the data"""

import pandas as pd

from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractMiddlePart, ExtractRawSentence

data = pd.read_csv('data/do-not-touch/held-out-data.csv')
embedding_size = 300

middle_part = ExtractMiddlePart().transform(data)
infersent = InfersentFeature(initialize_infersent(middle_part)).transform(middle_part)

embedding = []
for e in infersent:
    embedding.append(e.tolist())

data['pre_path_middle'] = ExtractMiddlePart(processing='replace_dist', rep_a='Objecta', rep_b='Objectb').transform(data)
data['pre_path_full'] = ExtractRawSentence(processing='replace_dist', rep_a='Objecta', rep_b='Objectb').transform(data)
data['embedding_middle_part'] = embedding
data.to_csv('data/do-not-touch/held-out-data-pre.csv', index=False)
