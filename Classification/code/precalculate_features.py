# adds the sentence embeddings from infersent to the data frame so that they don't have to be calculated multiple times

import pandas as pd
import numpy as np

from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractMiddlePart, ExtractRawSentence

data = pd.read_csv('data/data.csv')
embedding_size = 300
#restricted_embeddings = pd.read_csv('data/paths-restricted.csv')
#unrestricted_embeddings = pd.read_csv('data/paths-unrestricted.csv')

#data['restricted_path'] = data.apply(lambda row: [], axis=1)
#data['unrestricted_path'] = data.apply(lambda row: [], axis=1)
#data['restricted_path_embedding'] = data.apply(lambda row: np.zeros((embedding_size, 1), axis=1))
#data['unrestricted_path_embedding'] = data.apply(lambda row: np.zeros((embedding_size, 1), axis=1))

middle_part = ExtractMiddlePart().transform(data)
infersent = InfersentFeature(initialize_infersent(middle_part)).transform(middle_part)

embedding = []
for e in infersent:
    embedding.append(e.tolist())


data['pre_path_middle'] = ExtractMiddlePart(processing='replace_dist', rep_a='Objecta', rep_b='Objectb').transform(data)
data['pre_path_full'] = ExtractRawSentence(processing='replace_dist', rep_a='Objecta', rep_b='Objectb').transform(data)
data['embedding_middle_part'] = embedding
data.to_csv('data/data_if.csv',index=False)
