import json




def load(file):
    
    sentences = []
    already = set()
    idx = 0
    with open(file, 'r') as f:
        data = json.load(f)
        print(len(data))
        for item in data:
            a,b, sentence, with_marker = ((item['a'], item['b'], item['sentence'], item['without-marker']))
            if sentence.lower() not in already and is_valid([a, b], sentence):
                sentences.append('{}\t{}\t{}\t{}\t{}'.format(idx,sentence,a,b, with_marker))
                idx+=1
                already.add(sentence.lower())
    return sentences


def is_valid(words, sentence):
    count = 0
    try:
        for word in words:
            count += sentence.lower().count(word.lower())
        return count == 2
    except Exception as e:
        return False


typ = 'comp-sci'
setup = 'next_{}_max_{}'.format(3,25)

file_name = '../final-data/{}/{}/sentences-compsci'.format(typ,setup,typ)
#file_name = '../final-data/wordnet/sentences-wordnet'
sentences = load('{}.json'.format(file_name))
print(len(sentences))

with open('../final-data/{}/{}/{}-sentences-only.tsv'.format(typ,setup,typ), 'w') as out:
    out.write('id\tsentence\ta\tb\twithout_marker\n')
    for sentence in sentences:
        out.write(sentence+'\n')
