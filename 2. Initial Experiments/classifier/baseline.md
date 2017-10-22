CountVectorizer over whole sentence

=== LogisticRegression (Count)===
F1 68.52
             precision    recall  f1-score   support

A_GREATER_B       0.38      0.24      0.30        37
 A_LESSER_B       0.27      0.12      0.17        25
    NO_COMP       0.79      0.91      0.84       174

avg / total       0.67      0.72      0.69       236
 [[  9   2  26]
 [  5   3  17]
 [ 10   6 158]]



=== LinearSVC (Count)===
F1 65.25
             precision    recall  f1-score   support

A_GREATER_B       0.26      0.24      0.25        37
 A_LESSER_B       0.24      0.16      0.19        25
    NO_COMP       0.78      0.83      0.80       174

avg / total       0.64      0.67      0.65       236
 [[  9   3  25]
 [  6   4  15]
 [ 20  10 144]]


=== PassiveAggressiveClassifier (Count)===
F1 67.41
             precision    recall  f1-score   support

A_GREATER_B       0.31      0.30      0.30        37
 A_LESSER_B       0.31      0.16      0.21        25
    NO_COMP       0.79      0.85      0.82       174

avg / total       0.66      0.69      0.67       236
 [[ 11   2  24]
 [  6   4  15]
 [ 19   7 148]]


Process finished with exit code 0
