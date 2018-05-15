"""
This script tests all features on the held-out data.
"""

import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from xgboost import XGBClassifier

from features.contains_features import ContainsPos
from features.mean_embedding_feature import MeanWordEmbedding
from features.misc_features import SelectDataFrameColumn, POSTransformer
from transformers.data_extraction import ExtractMiddlePart
from util.data_utils import get_misclassified
from util.graphic_utils import print_confusion_matrix, plot
from util.misc_utils import get_logger
from util.precalculate_features import precalculate_embedding

MIDDLE_16 = 'middle_paths_unrestricted_16'
FULL_4 = 'full_paths_original_4'


def add_embeddings_to_df(df, embedding_file, column_name):
    embeddings = pd.read_csv(embedding_file)
    embeddings['embedding'] = embeddings.apply(lambda row: json.loads(row['embedding']), axis=1)
    df[column_name] = df.apply(lambda row: embeddings[embeddings.id == row['id']].embedding.values.tolist()[0], axis=1)
    return df


logger = get_logger('heldout_jl')

# adding the precalculated path embeddings and sentence embeddings
_train_data = pd.read_csv('data/data.csv')
train_data = add_embeddings_to_df(_train_data, '../all_data_files/extras/combi_full_paths_original_4.csv', FULL_4)
train_data = add_embeddings_to_df(_train_data, '../all_data_files/extras/combi_middle_paths_unrestricted_16.csv', MIDDLE_16)
train_data = precalculate_embedding(train_data)

_test_data = pd.read_csv('data/do-not-touch/held-out-data.csv')
test_data = add_embeddings_to_df(_test_data, '../all_data_files/extras/combi_full_paths_original_4.csv', FULL_4)
test_data = add_embeddings_to_df(_test_data, '../all_data_files/extras/combi_middle_paths_unrestricted_16.csv', MIDDLE_16)
test_data = precalculate_embedding(test_data)

print('Training Data Statistics: \n{}\n'.format(train_data.most_frequent_label.value_counts()))
print('Test Data Statistics: \n{}'.format(test_data.most_frequent_label.value_counts()))

feature_unions = [
    FeatureUnion([('full_paths_original_4', make_pipeline(SelectDataFrameColumn(FULL_4)))]),
   FeatureUnion([('middle_paths_unrestricted_16', make_pipeline(SelectDataFrameColumn(MIDDLE_16)))]),
 #   FeatureUnion([('Bag-Of-Words', make_pipeline(ExtractMiddlePart(), CountVectorizer()))]),
 #   FeatureUnion([('InferSent', make_pipeline(SelectDataFrameColumn('embedding_middle_part')))]),
 #   FeatureUnion([('Word Embedding', make_pipeline(ExtractMiddlePart(), MeanWordEmbedding()))]),
 #   FeatureUnion([('POS n-grams', make_pipeline(ExtractMiddlePart(), POSTransformer(), TfidfVectorizer(max_features=500, ngram_range=(2, 4))))]),
 #   FeatureUnion([('Contains JJR', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR')))]),
]


def run_experiment(binary):
    idx = 0
    idx_mc = 0
    result_frame = pd.DataFrame(columns=['feature', 'class', 'f1', 'precision', 'recall'])
    miss = pd.DataFrame(columns=['id', 'caption', 'sentence', 'object_a', 'object_b', 'predicted', 'gold'])
    labels = ['ARG', 'NONE'] if binary else ['BETTER', 'WORSE', 'NONE']
    for union in feature_unions:

        if binary:
            train_data['most_frequent_label'] = train_data.apply(lambda row: 'NONE' if row.most_frequent_label == 'NONE' else 'ARG', axis=1)
            test_data['most_frequent_label'] = test_data.apply(lambda row: 'NONE' if row.most_frequent_label == 'NONE' else 'ARG', axis=1)

        caption = union.transformer_list[0][0]
        logger.info(caption)
        classifier = XGBClassifier(n_jobs=8, n_estimators=1000)
        pipeline = make_pipeline(union, classifier)
        fitted = pipeline.fit(train_data, train_data['most_frequent_label'].values)
        predicted = fitted.predict(test_data)
        report = classification_report(test_data['most_frequent_label'].values, predicted, labels=labels, digits=2)
        matrix = confusion_matrix(test_data['most_frequent_label'].values, predicted, labels=labels)
        logger.info(report)
        logger.info(matrix)

        # build the graphics
        print_confusion_matrix('jl_heldout_{}_{}'.format(caption, binary), matrix, labels)

        result_frame.loc[idx] = [caption, 'Overall', f1_score(test_data['most_frequent_label'].values, predicted, average='weighted'), precision_score(test_data['most_frequent_label'].values, predicted, average='weighted'),
                                 recall_score(test_data['most_frequent_label'].values, predicted, average='weighted')]
        idx += 1
        for label in labels:
            result_frame.loc[idx] = [caption, label, f1_score(test_data['most_frequent_label'].values, predicted, average='weighted', labels=[label]),
                                     precision_score(test_data['most_frequent_label'].values, predicted, average='weighted', labels=[label]),
                                     recall_score(test_data['most_frequent_label'].values, predicted, average='weighted', labels=[label])]
            idx += 1

        for _id, sentence, a, b, predicted, gold in get_misclassified(predicted, test_data):
            miss.loc[idx_mc] = [_id, caption, sentence, a, b, predicted, gold]
            idx_mc += 1
        #miss.to_csv('missclassified/heldout__{}_binary_{}.csv'.format(caption, binary), index=False)

        logger.info("\n\n============\n\n")
    result_frame.to_csv('graphics/data/jl_heldout_results_{}.csv'.format(binary), index=False)


#run_experiment(False)
run_experiment(True)
