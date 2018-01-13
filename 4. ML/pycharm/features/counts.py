from .base_feature import BaseFeature


class Length(BaseFeature):
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


class NECount(BaseFeature):
    """number of entities per sentence"""
    def transform(self, documents):
        result = []
        for _doc in documents:
            doc = Length.nlp(_doc)
            result.append(len(list(doc.ents)))
        return super(NECount, self).reshape(result)


class NounChunkCount(BaseFeature):
    """number of noun chunks per sentence"""
    def transform(self, documents):
        result = []
        for _doc in documents:
            doc = Length.nlp(_doc)
            result.append(len(list(doc.noun_chunks)))
        return super(NounChunkCount, self).reshape(result)
