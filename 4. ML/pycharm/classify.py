import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline, FeatureUnion
from xgboost import XGBClassifier

from features.contains_features import ContainsPos
from features.count_features import NounChunkCount, NamedEntitiesByCategory, PunctuationCount
from features.mean_embedding_feature import MeanWordEmbedding
from features.misc_features import PositionOfObjects
from features.ngram_feature import NGramFeature
from infersent.infersent_feature import initialize_infersent, InfersentFeature
from transformers.data_extraction import ExtractMiddlePart, ExtractRawSentence
from transformers.n_gram_transformers import NGramTransformer
from util.data_utils import load_data, k_folds
from util.misc_utils import latex_table, get_logger, res_table
from util.ngram_utils import get_all_ngrams
from pprint import pprint, pformat

LABEL = 'most_frequent_label'

classifier_pattern = """
----------------------
- {}
----------------------"""
feature_name_pattern = """
========================================================================================

======================
- {}
======================"""

logger = get_logger('xgb_test_it3')

classifiers = [XGBClassifier(n_jobs=8, n_estimators=100)]
# classifiers = [RidgeClassifier(), XGBClassifier(), LinearSVC(), SVC(), SGDClassifier(), GaussianNB(),
#               KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
#               LogisticRegression(), ExtraTreesClassifier()]

n_gram_cache = {}


def n_gram_pipeline(n, extractor, processing=None, filter_punct=True, min_freq=1):
    def _n_gram_pipeline(train):
        ngram_base = extractor(processing=processing).transform(train)
        unigrams = get_all_ngrams(ngram_base, n, min_freq=min_freq, filter_punct=filter_punct)
        return [extractor(processing=processing), NGramTransformer(n, min_freq=min_freq, filter_punct=filter_punct),
                NGramFeature(unigrams, n=n)]

    return _n_gram_pipeline


def infersent_pipeline(extractor, processing=None):
    def _infersent_pipeline(train):
        raw_text = extractor(processing=processing).transform(train)
        infersent_model = initialize_infersent(raw_text)
        return [extractor(processing=processing), InfersentFeature(infersent_model)]

    return _infersent_pipeline


feature_builder = [

    ('TF-IDF', lambda train: [ExtractMiddlePart(), TfidfVectorizer()], 'middle part, unigrams'),
    ('TF-IDF', lambda train: [ExtractRawSentence(), TfidfVectorizer()], 'whole sentence, unigrams'),
    ('TF-IDF', lambda train: [ExtractMiddlePart(), TfidfVectorizer(ngram_range=(2, 2))], 'middle part, bigrams'),
    ('TF-IDF', lambda train: [ExtractRawSentence(), TfidfVectorizer(ngram_range=(2, 2))], 'whole sentence, bigrams'),
    ('TF-IDF', lambda train: [ExtractMiddlePart(), TfidfVectorizer(ngram_range=(3, 3))], 'middle part, trigrams'),
    ('TF-IDF', lambda train: [ExtractRawSentence(), TfidfVectorizer(ngram_range=(3, 3))], 'whole sentence, trigrams'),

    ('Sentence Embedding', infersent_pipeline(ExtractMiddlePart), 'middle part'),
    ('Sentence Embedding', infersent_pipeline(ExtractRawSentence), 'whole sentence'),

    ('Mean Word Embedding', lambda train: [ExtractRawSentence(), MeanWordEmbedding()], 'whole sentence'),
    ('Mean Word Embedding', lambda train: [ExtractMiddlePart(), MeanWordEmbedding()], 'middle part'),

    ('Unigrams', n_gram_pipeline(1, ExtractRawSentence), 'whole sentence, binary'),
    ('Unigrams', n_gram_pipeline(1, ExtractMiddlePart), 'middle part, binary'),

    ('Bigrams', n_gram_pipeline(2, ExtractRawSentence), 'whole sentence, binary'),
    ('Bigrams', n_gram_pipeline(2, ExtractMiddlePart), 'middle part, binary'),

    ('Trigrams', n_gram_pipeline(3, ExtractRawSentence), 'whole sentence, binary'),
    ('Trigrams', n_gram_pipeline(3, ExtractMiddlePart), 'middle part, binary'),

    ('Contains JJR', lambda train: [ExtractRawSentence(), ContainsPos('JJR')], 'whole sentence'),
    ('Contains JJR', lambda train: [ExtractMiddlePart(), ContainsPos('JJR')], 'middle part'),

    ('Contains JJS', lambda train: [ExtractRawSentence(), ContainsPos('JJS')], 'whole sentence'),
    ('Contains JJS', lambda train: [ExtractMiddlePart(), ContainsPos('JJS')], 'middle part'),

    ('Contains RBR', lambda train: [ExtractRawSentence(), ContainsPos('RBR')], 'whole sentence'),
    ('Contains RBR', lambda train: [ExtractMiddlePart(), ContainsPos('RBR')], 'middle part'),

    ('Contains RBS', lambda train: [ExtractRawSentence(), ContainsPos('RBS')], 'whole sentence'),
    ('Contains RBS', lambda train: [ExtractMiddlePart(), ContainsPos('RBS')], 'middle part'),

]


def run_classification(data, labels):
    by_score = []
    res_list = []
    for _builder in feature_builder:
        name, builder, comment = _builder
        logger.info(feature_name_pattern.format(name.upper() + " " + comment))
        for classifier in classifiers:

            logger.info(classifier_pattern.format(classifier))

            res = []
            f1_overall = 0;
            for train, test in k_folds(5, data):
                try:
                    steps = builder(train) + [classifier]
                    pipeline = make_pipeline(*steps)
                    fitted = pipeline.fit(train, train[LABEL].values)
                    predicted = fitted.predict(test)
                    logger.info(classification_report(test[LABEL].values, predicted, labels=labels))
                    f1 = f1_score(test[LABEL].values, predicted, average='weighted',
                                  labels=labels)
                    f1_overall += f1

                    res.append((f1_score(test[LABEL].values, predicted, average='weighted',
                                         labels=labels), (test[LABEL].values, predicted)))
                    now = datetime.datetime.now()
                    logger.info("{}:{}:{}\n\n".format(now.hour, now.minute, now.second))
                except Exception as e:
                    logger.info(e)
                    logger.info('Fail for {}'.format(type(classifier)).upper())
            res = sorted(res, key=lambda x: x[0])
            res_d = {'name': name, 'comment': comment, 'worst': res[0][0], 'avg': res[2][0], 'best': res[-1][0]}
            res_list.append(res_d)
            logger.info('OVERALL F1 {}'.format(f1_overall / 5.0))
            by_score.append((f1_overall / 5.0, '{} {}'.format(type(classifier), name)))
            try:
                logger.info(latex_table([res[0][1]] + [res[2][1]] + [res[4][1]], 'cap'))
            except Exception as e:
                logger.info("No table")
    res_table(res_list, logger)
    logger.info(pformat(sorted(by_score, key=lambda x: x[0], reverse=True)))


logger.info('# THREE CLASSES')
_data = load_data('data.csv', binary=False)
run_classification(_data, ['BETTER', 'WORSE', 'NONE'])

# logger.info('\n\nx--------------------------------------------\n\n')
# logger.info('# BINARY CLASSES')
# _data_bin = load_data('data.csv', min_confidence=0, binary=True)

# run_classification(_data_bin, ['ARG', 'NONE'])
