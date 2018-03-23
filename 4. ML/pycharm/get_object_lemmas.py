import pandas as pd
import spacy

if __name__ == '__main__':
    nlp = spacy.load('en')
    vocab = set()
    data = pd.read_csv('data/data.csv')

    objects = list(data['object_a'].values) + list(data['object_b'].values)
    objects = [o.replace(' ','SPACE').replace('-','DASH') for o in objects]

    for object in objects:
        p = list(nlp(object).sents)[0]
        lemmas = [t.lemma_ for t in p]
        l = ''.join(lemmas)
        vocab.add(l)

    with open('vocab.txt', 'w') as f:
        for w in sorted(list(vocab)):
            print('{}'.format(w), file=f)
