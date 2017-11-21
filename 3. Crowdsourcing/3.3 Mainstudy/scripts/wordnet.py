from nltk.corpus import wordnet as wn
import nltk
import pandas as pd


df = pd.read_csv('data/things.csv')



def get_antonym(term):
    synset = wn.synsets(term)
    for syn in synset:
        for l in syn.lemmas():
            if l.antonyms():
                print('.')
                return l.antonyms()[0].name()
    return ''


df['antonym'] = df.apply(lambda row: get_antonym(row['noun']), axis=1)

df.to_csv('foo2.csv')

#df = pd.read_csv('foo.csv')

#df[df['antonym'].notnull()].to_csv('random.csv', index=False)