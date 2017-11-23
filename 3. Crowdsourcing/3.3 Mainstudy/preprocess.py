import json

def load(file):
    sentences = []
    with open(file, 'r') as f:
        data = json.load(f)
        print(len(data))
        for item in data:
            a,b, sentence = ((item['a'], item['b'], item['sentence']))
            if is_valid([a, b], sentence):
                sentences.append(sentence)
    return sentences


def is_valid(words, sentence):
    
    count = 0
    for word in words:
        count += sentence.lower().count(word.lower())
    return True


sentences = load('bw-sentences-compsci.json')
print(len(sentences))

with open('sentence-compsci.txt', 'w') as out:
    for sentence in sentences:
        out.write(sentence+'\n')
