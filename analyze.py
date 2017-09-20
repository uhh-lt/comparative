import os
import json
from datetime import datetime
import getch
from random import sample

sentences = set()
hit = []
miss = []
skipped = []

def read():
    file_list = os.listdir('results')
    file_list.remove('.DS_Store')
    for f in file_list:
        try:
            with open('results/' + f) as data:
                if 'json' in f:
                        j = data.readlines()[0]
                        hits= json.loads(j)['result']['hits']['hits']
                        for hit in hits:
                            sentences.add(hit['_source']['text'])
        except Exception:
            pass


def write(lst, name):
    file_name ='analyze/'+ str(datetime.now().strftime('%d-%m_%H')) + '_' + name +'.txt'
    with open(file_name, 'w') as data:
        for line in lst:
            data.write(line+'\n')
    return file_name

def prompt():
    s_list = list(sentences)
    s_list.sort(key=lambda a: len(a))
    for i, sentence in enumerate(s_list):
        print('({}/{}) {}'.format(i, len(sentences), sentence))
        print('(A)ccept | (D)ecline | (S)kip')
        choice = getch.getch()
        if choice.lower() == 'a':
            hit.append(sentence)
        elif choice.lower() == 'd':
            miss.append(sentence)
        else:
            skipped.append(sentence)
    write(hit, 'hits')
    write(miss, 'miss')
    write(skipped, 'skipped')
    write(['Hits: {}'.format(len(hit)), 'Miss: {}'.format(len(miss)), 'Skipped: {}'.format(len(skipped)), 'Hit/Sum {}'.format(len(hit)/len(s_list)),], 'stats')




read()
write(sentences, 'raw')
prompt()