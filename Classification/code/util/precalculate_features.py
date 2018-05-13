"""Adds the prepared sentence for the path embeddings and the sentence embeddings (InferSent) to the data"""

from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractMiddlePart, ExtractRawSentence


def precalculate_embedding(data):
    middle_part = ExtractMiddlePart().transform(data)
    infersent = InfersentFeature(initialize_infersent(middle_part)).transform(middle_part)

    embedding = []
    for e in infersent:
        embedding.append(e.tolist())

    data['embedding_middle_part'] = embedding
    return data


def prepare_for_paths(data):
    data['pre_path_middle'] = ExtractMiddlePart(processing='replace_dist', rep_a='Objecta', rep_b='Objectb').transform(data)
    data['pre_path_full'] = ExtractRawSentence(processing='replace_dist', rep_a='Objecta', rep_b='Objectb').transform(data)
    return data
