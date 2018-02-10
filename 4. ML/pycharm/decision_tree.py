import subprocess

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data.data import *
from features.feature_sets import *
from infersent.infersent_feature import initialize_infersent
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion

labels = ['BETTER', 'WORSE',  'NONE']


CUE_WORDS_WORSE = ["worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior",
                   "mediocre"]

data = load_data('train-data.csv', min_confidence=0)
infersent_model = initialize_infersent(data['raw_text'].values)

features = [
    #('contains-worse', make_pipeline(ExtractMiddlePart(), ContainsWord(['than', 'but']))),
   # ('contains-a', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_WORSE))),

    ('jjr-m', make_pipeline(ExtractMiddlePart(), ContainsPos('JJR'))),
   # ('jjr-s', make_pipeline(ExtractMiddlePart(), ContainsPos('JJS'))),
   ('rbr', make_pipeline(ExtractMiddlePart(), ContainsPos('RBR'))),
   # ('rbs', make_pipeline(ExtractMiddlePart(), ContainsPos('RBS'))),

    #('infersent-m', make_pipeline(ExtractMiddlePart(processing=None), InfersentFeature(infersent_model))),
]
#features = get_all(data, infersent_model)

train, test = train_test_split(data, stratify=data['label'].values)

tree = DecisionTreeClassifier(min_samples_leaf=5)
pipeline = make_pipeline(FeatureUnion(features), tree)

print("fit")
fitted = pipeline.fit(train,
                      train['label'].values)
print("predict")
predictions = fitted.predict(test)

print(classification_report(test['label'].values, predictions, labels=labels))
print(confusion_matrix(test['label'].values, predictions, labels=labels))



print("export")
export_graphviz(
    tree,
    out_file='tree.dot',
    feature_names=get_feature_names(fitted),
    class_names=labels,
    rounded=True,
    filled=True, leaves_parallel=True
)

subprocess.call(['dot', '-Tpdf', 'tree.dot', '-o' 'tree.pdf'])

