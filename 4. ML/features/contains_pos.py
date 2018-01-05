from .base_feature import BaseFeature

class ContainsPos(BaseFeature):

    def test(self):

        print([t.tag_ for t in   BaseFeature.nlp("This is better than that")])
