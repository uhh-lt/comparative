import json
import multiprocessing
import requests
import logging
import numpy
from string import Template
from itertools import combinations

logging.basicConfig(level=logging.INFO)

QUALIFIERS = [
    'better than', 'worse than', 'inferior to', 'superior to', 'prefer'
]

MARKERS = [
    'because', 'since', 'as long as', 'as things go', 'cause of',
    'by reaso of', 'virtue of', 'considering', 'due to', 'for the reason that',
    'sake of', 'as much as', 'view of', 'thanks to', 'since', 'therefor',
    'thus'
]

BASE_URL = 'https://9d0fec4462c1b8723270b0099e94777e.europe-west1.gcp.cloud.es.io:9243/commoncrawl/sentence'
SEARCH_URL = BASE_URL + '/_search'
COUNT_URL = BASE_URL + '/_count'

USER = 'elastic'
PWD = 'yvmONIpMhHdp96ZpsxDKbPy4'


def worker_logger(name):
    """create a logger for worker threads"""
    formatter = logging.Formatter('%(asctime)s - %(name)s  - %(message)s')

    file_handler = logging.FileHandler('worker_{}.log'.format(name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger('Worker ' + str(name))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def fileToList(file):
    """read data file"""
    with open(file) as data:
        array = data.readlines()[0]
        lst = json.loads(array)
        return list(map(lambda entry: entry['label'], lst))


def count(object_a, object_b):
    """check if the combination a+b exists in the index"""
    es_query = Template(
        '{ "query": { "match": { "text": {"query": "$object_a $object_b", "operator": "and" } } } }'
    )
    resp = requests.get(
        COUNT_URL,
        params={
            'source': es_query.substitute(
                object_a=object_a, object_b=object_b)
        },
        auth=(USER, PWD))
    if resp.status_code == 200:
        return int(resp.json()['count'])
    else:
        resp.raise_for_status()


def query(object_a, object_b, prop):
    """check if a, b property, qualifier and marker combination exists"""
    es_query = Template(
        """{ "query" : { "match": { "text": {"query": "'$object_a' AND '$object_b' 
        AND $MARKERS AND $qualifier", "minimum_should_match": "75%"}}}}""")
    resp = requests.get(
        SEARCH_URL,
        params={
            'source':
            es_query.substitute(
                object_a=object_a,
                object_b=object_b,
                MARKERS='( {} )'.format(' OR '.join(MARKERS)),
                qualifier='( {} )'.format(' OR '.join(QUALIFIERS)))
        },
        auth=(USER, PWD))

    if resp.status_code == 200:
        result = resp.json()['hits']['hits']
        text = []
        for hit in result:
            text.append(hit['_source']['text'])
            return text
    else:
        resp.raise_for_status()


def worker(name, partition, props):
    """query es"""
    logger = worker_logger(str(name))
    for object_a, object_b in partition:
        logger.info('Check {} + {}'.format(object_a, object_b))
        co_occ = count(object_a, object_b)
        if co_occ > 0:
            for prop in props:
                logger.info('{} co-occurences of {} + {}'.format(co_occ, object_a, object_b))
                hits = query(object_a, object_b, prop)
                logger.info('{} hits for {}'.format(len(hits), prop))
                with open('res_worker_{}.txt'.format(name), 'a') as f:
                    f.write(hits)
        else:
            logger.info('No co-occurence')


def main():
    objects = fileToList('arg/obj/movie.json')
    props = fileToList('arg/prop/movie.json')
    objects_cross = numpy.array(list(combinations(objects, 2)))
    partitions = numpy.array_split(objects_cross, 8)
    print(len(objects_cross))
    jobs = []

    for index, partition in enumerate(partitions):
        p = multiprocessing.Process(
            target=worker, args=(index, partition, props))
        jobs.append(p)
      #  p.start()


if __name__ == '__main__':
    main()
