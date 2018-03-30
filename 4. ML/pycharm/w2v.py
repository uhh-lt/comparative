import pandas as pd
from gensim.models.word2vec import Word2Vec

paths = pd.read_csv('data/dependency_paths.csv')

tolist = paths.path.values.tolist()
tl = [a.split(' ') for a in tolist]

foo = set()
for t in tl:
    for f in t:
        foo.add(f)

model = Word2Vec(sentences=tl,min_count=1,size=300)
a = model.wv.vocab
print('{} {}'.format(len(a), len(foo)))
model.save('dependency_w2v_gensim')
