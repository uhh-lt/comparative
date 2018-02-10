from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from xgboost import XGBClassifier

from data.data import load_data
from features.contains import ContainsPos
from infersent.infersent_feature import initialize_infersent, InfersentFeature
from transformers.data_extraction import ExtractMiddlePart

data = load_data('train-data.csv', min_confidence=0, binary=False)
data['label'] = data.apply(
    lambda row: row['label'] if row['label'] != 'OTHER' else 'NONE', axis=1)

classifier = XGBClassifier()

infersent_model = initialize_infersent(data['raw_text'].values)

feature_list = [
    ('jjr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),
    ('infersent-m', make_pipeline(ExtractMiddlePart(processing=None), InfersentFeature(infersent_model))),
]

param_grid = {
    # 'featureunion__jjr-m__extractmiddlepart__processing': [None, 'replace', 'replace_dist', 'remove']
    # 'featureunion__infersent-m__extractmiddlepart__processing': [None, 'replace', 'replace_dist', 'remove']
    #'xgbclassifier__booster': ['gblinear'],
    'xgbclassifier__learning_rate': [1,0.0001,0.001,0.5],

}

pipeline = make_pipeline(FeatureUnion(feature_list), classifier)
print(pipeline.get_params())

gridsearch = GridSearchCV(pipeline, param_grid=param_grid, cv=StratifiedKFold(n_splits=3, random_state=42),
                          scoring="f1_weighted", verbose=10)
gridsearch.fit(data, data['label'])

print('=== Best Parameters ===')
print(gridsearch.best_params_)
