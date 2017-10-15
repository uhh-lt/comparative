"""count co-occurences of two objects given a class; needed to find best pairs for bootstrapping"""
import argparse
import grequests
import sys
from random import shuffle
from collections import defaultdict
from pprint import pprint
from itertools import combinations
sys.path.insert(0, '../')
from HelperModule.utils import file_to_list, jsonify_no

ELASTIC_SEARCH_URL = 'http://localhost:9222/commoncrawl/sentence/_count'
REQUEST_THREADS = 5


PROG_LANG = ['Java', 'Ruby', 'Python', 'JavaScript',
'PHP', 'Swift', 'Scala', 'Perl', 'SQL', 'Haskell', 'Erlang'
'Lua', 'Visual Basic', 'Fortran', 'Rust', 'Go']

CD = ['cat', 'dog']

OS = ['Windows', 'OS X', 'MacOS', 'Ubuntu', 'Linux', 'Debian']

def count(terms):
    """check if a, b property, qualifier and marker combination exists"""
    headers = {
        'Content-Type': 'application/json',
        'x-terms' : str(terms)
    }
    print('Process: {}'.format(terms))
    query = """
    {{
        "query" : {{
            "bool": {{
                "filter": [
                    {{
                        "query_string": {{
                            "default_field" : "text",
                            "query" :  "(better OR worse OR superior OR inferior) AND {}"
                        }}
                    }}
                ]
            }}
        }}
    }}
    """
    print(query.format(jsonify_no(terms, joiner=' AND ')))
    return grequests.post(
        ELASTIC_SEARCH_URL,
        data=query.format(jsonify_no(terms, joiner=' AND ')),
        headers=headers,
        timeout=60)


def main():
    objects = OS
    shuffle(objects)
    print(len(objects))
    d = defaultdict(list)
    combi = combinations(objects,2)
    requests = []
    for c in combi:
        requests.append(count(c))

    response = grequests.map(
        requests,
        size=REQUEST_THREADS,
        exception_handler=lambda x, y: print(x,y))

    for answer in response:
        try:
            d[answer.json()['count']].append(answer.request.headers['x-terms'])
        except Exception as e:
            print(e)

    pprint(d)




if __name__ == '__main__':
    main()
