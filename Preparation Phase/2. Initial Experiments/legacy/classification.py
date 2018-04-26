import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, f1_score
from pandas import DataFrame
import getch
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer



def build_data_frame(file_name):
    rows = []
    with open(file_name, 'r') as data:
        for line in data:
            text, label = re.split(r'\t', line)
            rows.append({'text': text.strip(), 'label': label.strip()})
    data_frame = DataFrame(rows)
    return data_frame




def main():
    data = build_data_frame('2-all.labeled')

    pipeline = Pipeline([('features', FeatureUnion([('text', Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ]))])), ('clf', OneVsRestClassifier(LinearSVC()))])

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'].values,
        data['label'].values,
        test_size=0.2,
        random_state=0)

    pipeline.fit(X_train, y_train)
    s = pipeline.score(X_test, y_test)
    print(s)

    while (False):
        print("Input")
        sentence = [input()]
        print(pipeline.predict(sentence))


if __name__ == '__main__':
    main()