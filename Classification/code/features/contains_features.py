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

    def get_feature_names(self):
        return ['contains_pos_{}'.format(self.pos)]
