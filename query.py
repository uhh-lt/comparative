import json
import logging

from string import Template
from itertools import combinations
from random import shuffle, randint
from functools import reduce
from collections import defaultdict
from pprint import pprint as pp
from os.path import isfile
import grequests

logging.basicConfig(level=logging.INFO)

QUALIFIERS = [
    'better than', 'worse than', 'inferior to',
    'superior to', 'prefer'
]

MARKERS = [
    'because', 'since', 'as long as', 'as things go',
    'cause of', 'by reaso of', 'virtue of', 'considering',
    'due to', 'for the reason that', 'sake of',
    'as much as', 'view of', 'thanks to', 'since',
    'therefor', 'thus'
]



BASE_URL = 'https://9d0fec4462c1b8723270b0099e94777e.europe-west1.gcp.cloud.es.io:9243/commoncrawl/sentence'
SEARCH_URL = BASE_URL + '/_search'
COUNT_URL = BASE_URL + '/_count'

USER = 'elastic'
PWD = 'yvmONIpMhHdp96ZpsxDKbPy4'

COUNT_CACHE = {}

def match_phrase(phrase):
    return '{{ "match_phrase" : {{ "text": "{}"  }} }}'.format(phrase)

def query_string_or(words, min_match=0):
    quoted = ['\\"'+x+'\\"' for x in words]
    mms = ''
    if min_match > 0:
        mms = '"minimum_should_match" : {},'.format(min_match)
    return '{{ "query_string" : {{ {} "default_field": "text", "query" : "{}" }} }}'.format(mms,' OR '.join(quoted))


def worker_logger(name):
    """create a logger for worker threads"""
    formatter = logging.Formatter('%(asctime)s - %(name)s  - %(message)s')

    file_handler = logging.FileHandler('{}.log'.format(name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(str(name) + ' ')
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


GLOBAL_LOGGER = worker_logger('global')


def fileToList(file):
    """read data file"""
    with open(file) as data:
        array = data.readlines()[0]
        lst = json.loads(array)
        return list(map(lambda entry: entry['label'], lst))


def list_to_file(name, lst):
    with open(name, 'w') as data:
        for line in lst:
            data.write('{}\n'.format(line))


def query(query_string):
    """check if a, b property, qualifier and marker combination exists"""

    body = Template(
        '{ "query": { "bool": { "must": [ { "query_string": { "default_field": "text", "query": "\\"$object_a\\"" } }, { "query_string": { "default_field": "text", "query": "\\"$object_b\\"" } }, { "query_string": { "default_field": "text", "query": "$markers" } }, { "query_string": { "default_field": "text", "query": "$qualifier" } }, { "query_string": { "default_field": "text", "query": "\\"$prop\\"" } } ] } } }'
    )

    body2 = body.substitute(
        object_a=object_a,
        object_b=object_b,
        prop=prop,
        qualifier=' OR '.join(QUALIFIERS),
        markers=' OR '.join(MARKERS))

    headers = {'Content-Type': 'application/json'}
    return 0
    """
    resp = requests.post(
        SEARCH_URL, json=json.loads(body2), headers=headers, auth=(USER, PWD))


    if resp.status_code == 200:
        result = resp.json()['hits']['hits']
        text = []
        for hit in result:
            text.append(hit['_source']['text'])
            return text
    else:
        resp.raise_for_status()
        """


def count(objects):
    """checks if a sentence with all the objects exist"""
    obj_key = ' '.join(['\\"' + o + '\\"' for o in objects])
    es_query = ' {{ "query" : {{ "bool" : {{ "must": [ {} ] }} }} }}'.format(', '.join([match_phrase(x) for x in objects]))

    return grequests.get(
        COUNT_URL,
        params={'source': es_query},
        headers={'x-requested-object': obj_key.replace('\\"', '')},
        auth=(USER, PWD))


def count_all(obj_list, name):
    file_name = 'es/counts/{}.json'.format(name)
    dic = {}
    if isfile(file_name):
        with open(file_name, 'r') as cached:
            dic = json.load(cached)
            GLOBAL_LOGGER.info('Using count cache for {}'.format(name))

    else:
        dic = defaultdict(list)
        requests = []
        for obj in obj_list:
            requests.append(count([obj]))
        response = grequests.imap(
            requests, size=50, exception_handler=exception_handler)
        for value in response:
            obj = value.request.headers['x-requested-object']
            GLOBAL_LOGGER.info('Recieve {} {}'.format(obj, value.json()['count']))
            dic[value.json()['count']].append(obj)
        with open(file_name.format(name), 'w') as out:
            json.dump(dic, out)
    dic.pop('0', None)
    listed = []
    for l in dic.values():
        listed.extend(l)
    return listed


def exception_handler(req, exception):
    GLOBAL_LOGGER.info(req)
    GLOBAL_LOGGER.info(exception)


def main():
    object_cnt = 250
    object_name = 'actor'
    objects =  fileToList('arg/obj/{}.json'.format(object_name))
    properties = fileToList('arg/prop/{}.json'.format(object_name))[:5]
    shuffle(objects)

    obj_to_search = count_all(objects[:object_cnt], '{}{}'.format(object_name,object_cnt))
    test = """
    {{
        "query" : {{
            "bool": {{
                "must": [
                    {}
                ]
            }}
        }}
    }}
    """.format(', '.join([
        query_string_or(obj_to_search,  min_match=2)
       # query_string_or(properties, min_match=1),
       # query_string_or(MARKERS),
       # query_string_or(QUALIFIERS)
    ]))
    GLOBAL_LOGGER.info(test)


    #



if __name__ == '__main__':
    main()
