import re
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from pandas import DataFrame
import getch
from sklearn.model_selection import train_test_split


def build_data_frame(file_name):
    rows = []
    with open(file_name, 'r') as data:
        for line in data:
            text, label = re.split(r'\t', line)
            rows.append({'text' : text.strip(), 'label': label.strip()})
    data_frame = DataFrame(rows)
    return data_frame

def main():
    data = build_data_frame('1-combined.labeled')

    pipeline = Pipeline([('vectorizer', CountVectorizer(
        stop_words=['Python', 'Ruby', 'OS X', 'Windows', 'cat',
                    'dog'])), ('tfidf_transformer', TfidfTransformer()),
                         ('classifier', LinearSVC())])

    X_train, X_test, y_train, y_test = train_test_split(
    data['text'].values, data['label'].values, test_size=0.2, random_state=0)

    pipeline.fit(X_train, y_train)
    s = pipeline.score(X_test, y_test)
    print(s)

    """
    k_fold = KFold(n=len(data), n_folds=10)
    scores = []
    confusion = numpy.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])

    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['label'].values

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['label'].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions,average='micro')
        scores.append(score)

    print('Total classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)"""
    while(True):
        print("Input")
        sentence = [input()]
        print(pipeline.predict(sentence))



if __name__ == '__main__':
    main()