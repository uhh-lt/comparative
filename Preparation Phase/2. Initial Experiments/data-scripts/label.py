import sys
sys.path.insert(0, '../../Other')
import json
import getch
from datetime import datetime
from collections import defaultdict
from nltk.stem import PorterStemmer
from random import shuffle, sample

sys.path.insert(0, '../')

stemmer = PorterStemmer()


def normalize(s):
    puncts = '.?!/",;-\\:'
    for sym in puncts:
        s = s.replace(sym, '').replace('  ', ' ')
    return ''.join([stemmer.stem(x) for x in s.lower().strip()])


def read(file):
    n = []
    lines = []
    cnt = 0
    with open(file) as data:
        lines = [x['_source']['text'] for x in json.load(data)['hits']['hits']]
        shuffle(lines)
    return sorted(sample(lines,500), key=len)


def label(sentences, target_file):
    d = defaultdict(list)
    with open(target_file, 'w') as t:
        t.write('# START\n')
        for i, s in enumerate(sentences):
            print('{} ({}/{})'.format(s, i + 1, len(sentences)))
            print('(a)  >  \t (d)  <  \t (g)  =  \t (p) Out')
            su = len(d['>']) + len(d['<']) + len(d['=']) + len(d['o']) + len(
                d['skipped'])
            print('> {} | < {} | = {} | o {} | sum: {}'.format(
                len(d['>']), len(d['<']), len(d['=']), len(d['o']), su))
            choice = getch.getch()
            if choice is 'a':
                d['>'].append(s)
                t.write('{}\A_GREATER_B\n'.format(s))
                print(chr(27) + "[2J")
            elif choice is 'd':
                d['<'].append(s)
                t.write('{}\A_LESSER_B\n'.format(s))
                print(chr(27) + "[2J")
            elif choice is 'd':
                d['<'].append(s)
                t.write('{}\A_LESSER_B\n'.format(s))
                print(chr(27) + "[2J")
            elif choice is 'g':
                d['='].append(s)
                t.write('{}\tNO_COMP\n'.format(s))
                print(chr(27) + "[2J")
            elif choice is 'p':
                d['o'].append(s)
                t.write('{}\tOUT\n'.format(s))
                print(chr(27) + "[2J")
            elif choice is '9':
                d['skipped'].append(s)
                print(chr(27) + "[2J")


def main():
    """
    label(
        read('coke-pepsi.json'), '{}-coke-pepsi.labeled'.format(
            str((datetime.now().strftime('%m-%d-%H:%M')))))

    label(
        read('car-bike.json'), '{}-car-bike.labeled'.format(
            str((datetime.now().strftime('%m-%d-%H:%M')))))
    """

    examples = []
    with open('canon-nikon-sentences.txt', 'w') as f:
        for line in list(read('canon-nikon.json')):
            f.write(line+'\n')

    shuffle(examples)

 #   label(examples, '{}.labeled'.format(
 #           str((datetime.now().strftime('%m-%d-%H:%M')))))


if __name__ == '__main__':
    main()