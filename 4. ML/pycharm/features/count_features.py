from .base_feature import BaseFeature
from spacy.symbols import PERSON, ORG, GPE, LOC, PRODUCT, WORK_OF_ART, LANGUAGE, PERCENT, MONEY, ORDINAL, QUANTITY, \
    CARDINAL
from collections import OrderedDict
import numpy as np


class Length(BaseFeature):
    """expects any iterable, returns a vector with the lenght for each element"""

    def transform(self, documents):
        result = []
        for doc in documents:
            result.append(len(doc))
        return super(Length, self).reshape(result)


class PunctuationCount(BaseFeature):
    """number of punctuation characters per sentence"""

    def transform(self, documents):
        result = []
        for _doc in documents:
            doc = Length.nlp(_doc)
            counter = 0
            for token in doc:
                if token.is_punct:
                    counter += 1
            result.append(counter)
        return super(PunctuationCount, self).reshape(result)


class NEOverallCount(BaseFeature):
    """number of entities per sentence"""

    def transform(self, documents):
        result = []
        for _doc in documents:
            doc = Length.nlp(_doc)
            result.append(len(list(doc.ents)))
        return super(NEOverallCount, self).reshape(result)


class NamedEntitiesByCategory(BaseFeature):

    def __init__(self, types=[PERSON, ORG, GPE, LOC, PRODUCT, WORK_OF_ART, LANGUAGE, PERCENT, MONEY, ORDINAL, QUANTITY,
                              CARDINAL]):
        self.types = types

    def transform(self, documents):
        result = []
        for _doc in documents:
            entity_counts = {k: 0 for k in self.types}
            entity_counts = OrderedDict(sorted(entity_counts.items()))
            doc = NamedEntitiesByCategory.nlp(_doc)
            for ent in doc.ents:
                if ent.label in entity_counts:
                    entity_counts[ent.label] += 1
            result.append(list(entity_counts.values()))

            r = np.reshape(np.asarray(result), (len(result), -1))
        return result


class NounChunkCount(BaseFeature):
    """number of noun chunks per sentence"""

    def transform(self, documents):
        result = []
        for _doc in documents:
            doc = Length.nlp(_doc)
            result.append(len(list(doc.noun_chunks)))
        return super(NounChunkCount, self).reshape(result)
