#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from random import shuffle
from collections import defaultdict
from os.path import isfile
import grequests
from datetime import datetime

logging.basicConfig(level=logging.INFO)

QUALIFIERS = ['better than', 'is better']

MARKERS = ['because']

BASE_URL = 'https://9d0fec4462c1b8723270b0099e94777e.europe-west1.gcp.cloud.es.io:9243/commoncrawl/sentence'
SEARCH_URL = BASE_URL + '/_search'
COUNT_URL = BASE_URL + '/_count'

USER = 'elastic'
PWD = 'yvmONIpMhHdp96ZpsxDKbPy4'

COUNT_CACHE = {}


def jsonify(lst):
    quote = ['"{}"'.format((x.encode('utf-8').decode('utf-8'))) for x in lst]
    joined = ' OR '.join(quote)
    return json.dumps(joined)


def query_string_or(words, min_match=0):
    quoted = jsonify(words)
    mms = ''
    if min_match > 0:
        mms = '"minimum_should_match" : {},'.format(min_match)
    return '{{ "query_string" : {{ {} "default_field": "text", "query" : {} }} }}'.format(
        mms, quoted)


def worker_logger(name):
    """create a logger for worker threads"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(
        'logs/{}_query.log'.format(str(datetime.now().strftime('%m-%d-%H:%M'))))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


LOG = worker_logger('global')


def fileToList(file):
    """read data file"""
    with open(file, encoding='utf-8') as data:
        array = data.readlines()[0]
        lst = json.loads(array)
        return list(map(lambda entry: entry['label'], lst))


def list_to_file(name, lst):
    with open(name, 'w') as data:
        for line in lst:
            data.write('{}\n'.format(line))


def query(partition, query_string, concept):
    """check if a, b property, qualifier and marker combination exists"""
    headers = {
        'Content-Type': 'application/json',
        'x-requested-object': concept.replace('\\"', '')
    }

    return grequests.post(
        SEARCH_URL,
        data=query_string,
        headers=headers,
        timeout=60,
        auth=(USER, PWD))


def exception_handler(req, exception):
    LOG.fatal(req.kwargs)
    LOG.fatal(exception)


def partition_to_size(size, data):
    return [data[x:x + size] for x in range(0, len(data), size)]

def remove_stopwords(lst, stopwords):
    return [x for x in lst if x not in stopwords]


stopwords = [

]


def main():
    group_size = 100
    max_props = 25
    add_props = True
    add_markers = True
    requests = []
    LOG.info('Group size: {}'.format( group_size))
    LOG.info('Qualifiers: {}'.format( QUALIFIERS))
    if add_markers: LOG.info('Marker: {}'.format(MARKERS))
    if add_props: LOG.info('Property size: {}'.format(max_props))

    with open('arg/conceptList.txt', encoding='utf-8') as l:
        files = list(l)
        resp_counter = 0
        query_counter = 0
        for line in files:
            try:
                object_name = line.strip().replace(' ', '_')
                objects = fileToList('arg/obj/{}.json'.format(object_name))
                properties = remove_stopwords(fileToList(
                    'arg/prop/{}.json'.format(object_name)), stopwords)[:max_props]
                #LOG.info(properties)
                shuffle(objects)
                LOG.info(
                    'Start {} (size {})'.format(object_name, len(objects)))
                partitions = partition_to_size(group_size, objects)
                cnt = 0
                for partition in partitions:

                    parameters = [query_string_or(partition, min_match=2), query_string_or(QUALIFIERS)]
                    if add_markers:
                        parameters.append(query_string_or(MARKERS))
                    if add_props:
                        parameters.append(query_string_or(properties))

                    es_query = '{{ "query" : {{ "bool": {{ "filter": [ {} ] }} }} , "highlight" : {{ "pre_tags": ["**"], "post_tags" : ["**"], "fields" : {{ "text" : {{}} }} }} }}'.format(
                        ', '.join(parameters))
                    LOG.info(es_query)
                    LOG.info('Query for ({}) {}; Partition size {}'.format(
                        cnt, object_name, len(partition)))
                    cnt = cnt + 1
                    query_counter += 1
                    requests.append(     query(partition, es_query, str(cnt) + '_' + object_name))
                cnt = 0
                response = grequests.map(
                requests, size=25, exception_handler=exception_handler)
                requests = []
                for result in response:
                    resp_counter += 1
                    if result and result.status_code is 200 and result.json(
                    )['hits']['total'] > 0:
                        file_name = 'results/{}.json'.format(
                            result.request.headers['x-requested-object'])
                        LOG.info('{} succeeded'.format(result.request.headers[
                            'x-requested-object']))
                        with open(file_name, 'w') as out:
                            json.dump({'result': result.json()}, out)
                    else:
                        try:
                            result.raise_for_status()
                            LOG.info('No results for {}'.format(
                                result.request.headers['x-requested-object']))
                        except Exception as e:
                            LOG.fatal(
                                '{} failed: {}'.format(result.request.headers[
                                    'x-requested-object'], result.status_code))

            except IOError as e:
                pass
            except Exception as x:
                LOG.info(x)
        LOG.info('Responses {} | (Queries {})'.format(resp_counter, query_counter))


if __name__ == '__main__':
    main()
