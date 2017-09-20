import os
import json
from datetime import datetime
import argparse
import getch
from random import sample

sentences = set()
hit = []
miss = []
skipped = []

PARSER = argparse.ArgumentParser()
PARSER.add_argument('folder', help='result folder')


def read(folder):
    file_list = os.listdir(folder)
    for f in file_list:
        try:
            with open(folder+'/' + f) as data:
                if 'json' in f:
                    j = data.readlines()[0]
                    hits= json.loads(j)['result']['hits']['hits']
                    for hit in hits:
                        sentences.add(hit['_source']['text'])
        except Exception:
            pass


def write(lst, folder, name):
    file_name = folder + '/analyze/' + str(datetime.now().strftime('%d-%m_%H')) + '_' + name +'.txt'
    with open(file_name, 'w') as data:
        for line in lst:
            data.write(line+'\n')
    return file_name

def prompt(folder):
    limit = 150
    s_list = list(sentences)
    s_list.sort(key=lambda a: len(a))
    for i, sentence in enumerate(s_list[:limit]):
        print('({}/{}) {}'.format(i, limit, sentence))
        print('(A)ccept | (D)ecline | (S)kip')
        choice = getch.getch()
        if choice.lower() == 'a':
            hit.append(sentence)
        elif choice.lower() == 'd':
            miss.append(sentence)
        else:
            skipped.append(sentence)
    write(hit, folder, 'hits')
    write(miss, folder, 'miss')
    write(skipped, folder, 'skipped')
    write(['Hits: {}'.format(len(hit)), 'Miss: {}'.format(len(miss)), 'Skipped: {}'.format(len(skipped)), 'Hit/(hit+miss) {}'.format(len(hit)/(len(hit)+len(miss))),], folder, 'stats')



def main():
    args = PARSER.parse_args()
    folder = args.folder
    if not os.path.exists(folder+'/analyze'):
        os.makedirs(folder+'/analyze')
    read(folder)
    write(sentences, folder,'raw')
    prompt(folder)


if __name__ == '__main__':
    main()