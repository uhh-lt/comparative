from features.mean_wordembedding import *
from features.contains_pos import *
from features.ngram import *

from transformers.sentence_split import *

from pandas import DataFrame as df

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.dummy import DummyClassifier


def load_data(file_name, min_confidence=0.7):
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    return shuffle(frame, random_state=42)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits)
    for train_index, test_index in k_fold.split(data['raw_text'],
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def build_feature_union():
    mean_word_embedding = ('mwe',
                           Pipeline([('step-1', MeanWordEmbedding())]))
    contains_jjr = ('contains-jjr', Pipeline([('step-1', ContainsPos('JJR'))]))
    unigram = ('unigram',
               Pipeline([('pre', SentenceSplit()), ('step-1', NGram(1, data['raw_text'].values))]))
    bigram = ('bigram',
              Pipeline([('step-1', NGram(2, data['raw_text'].values))]))
    trigram =  ('trigram', make_pipeline(NGram(3, data['raw_text'].values)))

    return FeatureUnion([

        mean_word_embedding,
        #contains_jjr,
        #contains_jjs,
        #contains_nnp,
        unigram,
        #bigram
    ]
    )

# ------ Classification
labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
data = load_data('brands_500.csv')

def run_pipeline(estimator, feature_union=build_feature_union()):
    pipeline = Pipeline([('features', feature_union),  ('estimator',
                                                       estimator)])
    fitted = pipeline.fit(train['raw_text'].values,
                          train['label'].values)
    predictions = fitted.predict(test['raw_text'].values)

    report = classification_report(
        test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(
        test['label'].values, predictions, labels=labels)
    return report, conf


for train, test in split_data(2, data):
#    report_dummy, conf_dummy = run_pipeline(DummyClassifier(strategy='most_frequent'))
    report, conf = run_pipeline(LogisticRegression())
#    print('Baseline: Most Frequent Class\n', report_dummy)
    print('Result\n', report, conf)
