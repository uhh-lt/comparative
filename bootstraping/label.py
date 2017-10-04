import sys
import json
import getch
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '../')


def read(file):
    with open(file) as data:
        line = json.load(data)['hits']['hits']
        return [x['_source']['text'] for x in line]

def label(a,b, sentences, target_file):
    d = defaultdict(list)
    with open(target_file, 'w') as t:
        t.write('# COMPARING {} to {}\n\n\n'.format(a,b))
        for i, s in enumerate(sentences):
            print('{} ({}/{})'.format(s,i+1,len(sentences)))
            print('(a) {} > {} \t (d) {} < {} \t (g) {} = {} \t (p) Out'.format(a,b,a,b,a,b))
            su = len(d['>']) + len(d['<']) + len(d['=']) + len(d['o']) + len(d['skipped'])
            print('> {} | < {} | = {} | o {} | sum: {}'.format(len(d['>']), len(d['<']),len(d['=']), len(d['o']), su))
            choice = getch.getch()
            if choice is 'a':
                d['>'].append(s)
                t.write('{}\tBETTER\n'.format(s))
                print(chr(27) + "[2J")
            elif choice is 'd':
                d['<'].append(s)
                t.write('{}\tWORSE\n'.format(s))
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
    label('Ruby', 'Python',read('ruby-python.json'), '{}-ruby-python.labeled'.format(str((datetime.now().strftime('%m-%d-%H:%M')))))
    
    label('cat', 'dog',
          read('cat-dog.json'), '{}-cat-dog.labeled'.format(
              str((datetime.now().strftime('%m-%d-%H:%M')))))

    label('OS X', 'Windows',read('osx-windows.json'), '{}-osx-windows.labeled'.format(str((datetime.now().strftime('%m-%d-%H:%M')))))
    




if __name__ == '__main__':
    main()