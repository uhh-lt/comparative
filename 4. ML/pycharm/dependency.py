from util.data_utils import load_data
import spacy
from pprint import pprint

nlp = spacy.load('en')


def find_root(docu):
    for token in docu:
        if token.dep_ == "ROOT":
            return token


sentence = nlp('Python is better than Java and Ruby , isn\'t it?')

print(sentence)

def search(sentence, neighbor_f):
    vertex = [find_root(sentence)]
    stack = [find_root(sentence)]
    parent = {}
    seen = set(vertex)
    while not len(stack) == 0:
         v = stack.pop()
         for u in list(v.lefts) + list(v.rights):
              if not u in seen:
                  seen.add(u)
                  stack.append(u)
                  parent[u.text] = v.text
    return parent


#p_left = search(sentence, lambda x: x.lefts)
p_right = search(sentence, lambda x: x.rights)

def path_to(d, word):
    p = d[word]
    path = [p]
    while p != find_root(sentence).text:
        p = d[p]
        path.append(p)

    path.reverse()
    return path



print(path_to(p_right, 'Python'))
print(path_to(p_right, 'Ruby'))
