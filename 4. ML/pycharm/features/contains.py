from .base_feature import BaseFeature


class ContainsPos(BaseFeature):
    """Boolean feature; checks if the sentences contains the given POS"""

    def __init__(self, pos):
        self.pos = pos

    def transform(self, documents):
        result = []
        for doc in documents:
            pre = ContainsPos.nlp(doc)
            pos = [t.tag_ for t in pre]
            result.append(self.pos.upper() in pos)
        return super(ContainsPos, self).reshape(result)


class ContainsWord(BaseFeature):
    """boolean feature if the string contains this word"""

    def __init__(self, words):
        self.words = words

    def transform(self, documents):
        result = []
        for doc in documents:
            pre = ContainsPos.nlp(doc)
            words = [t.text for t in pre]
            tmp = []
            for w in self.words:
                tmp.append(w in words)
            result.append(tmp)
        return result
