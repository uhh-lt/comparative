from .base_feature import BaseFeature
import random

class Potato(BaseFeature):

    def transform(self, documents):
        result = []
        for doc in documents:
            result.append(random.gauss(10,1))
        return super(Potato, self).reshape(result)
