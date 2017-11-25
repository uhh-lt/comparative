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
    try:
        for word in words:
            count += sentence.lower().count(word.lower())
        return count == 2
    except Exception as e:
        return False


file_name = 'sentences-brand-list'
sentences = load('{}.json'.format(file_name))
print(len(sentences))

with open('{}-sentences-only.txt'.format(file_name), 'w') as out:
    for sentence in sentences:
        out.write(sentence+'\n')
