from sklearn.base import TransformerMixin, BaseEstimator


class ExtractRawSentence(TransformerMixin, BaseEstimator):

    def transform(self, dataframe):
        results = []
        for index, row in dataframe.iterrows():
            results.append(row['raw_text'])

        return results

    def fit(self, X, y):
        return self


    def get_feature_names(self):
        return 'FullExtractor'




class ExtractFirstPart(TransformerMixin, BaseEstimator):
    """returns all words before the first object"""

    def transform(self, dataframe):
        results = []
        for index, row in dataframe.iterrows():
            a, b, text = row['a'], row['b'], row['raw_text']
            a_index, b_index = text.index(a), text.index(b)
            if a_index < b_index:
                begin, end = a_index, b_index
            else:
                begin, end = b_index, a_index
            res = str(text[:begin])
            results.append(res)
        return results

    def fit(self, X, y):
        return self


    def get_feature_names(self):
        return 'FirstExtractor'


class ExtractLastPart(TransformerMixin, BaseEstimator):
    """returns all words after the second object"""

    def transform(self, dataframe):
        results = []
        for index, row in dataframe.iterrows():
            a, b, text = row['a'], row['b'], row['raw_text']
            if text.index(a) > text.index(b):
                res = text.split(a)[-1]
            else:
                res = text.split(b)[-1]
            results.append(res)
        return results

    def fit(self, X, y):
        return self


    def get_feature_names(self):
        return 'LastExtractor'


class ExtractMiddlePart(TransformerMixin, BaseEstimator):
    """returns all words between the first and the second object"""

    def transform(self, dataframe):
        results = []
        for index, row in dataframe.iterrows():
            a, b, text = row['a'], row['b'], row['raw_text']
            a_index, b_index = text.index(a), text.index(b)
            if a_index < b_index:
                begin, end = a_index + len(a), b_index
            else:
                begin, end = b_index + len(b), a_index
            res = text[begin:end]

            results.append(res)

        return results

    def fit(self, X, y):
        return self

    def get_feature_names(self):
        return 'MiddleExtractor'
