#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import logging
from datetime import datetime
from random import shuffle
from collections import defaultdict
from os.path import isfile
import grequests
from HelperModule import file_to_list, query_string_or


REQUEST_THREADS = 25


BASE_URL = 'https://9d0fec4462c1b8723270b0099e94777e.europe-west1.gcp.cloud.es.io:9243/commoncrawl/sentence'
SEARCH_URL = BASE_URL + '/_search'
COUNT_URL = BASE_URL + '/_count'

USER = 'elastic'
PWD = 'yvmONIpMhHdp96ZpsxDKbPy4'



PARSER = argparse.ArgumentParser(description='Query elastic search')
PARSER.add_argument('objects', type=int, help='objects to compare at once')
PARSER.add_argument('props', type=int, help='properties to compare at once')
PARSER.add_argument('--marker', metavar='-m', nargs='+', help='additional words to add to the query')
PARSER.add_argument(
    '--l',
    metavar='-l',
    type=int,
    help='if additional words are provided, how many of them must appear in the result (defaults to 1)')


def get_logger(folder, name):
    """create a logger for worker threads"""
    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(
        folder+'/{}_{}.log'.format(str(datetime.now().strftime('%m-%d-%H:%M')), name))
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



def partition_to_size(size, data):
    return [data[x:x + size] for x in range(0, len(data), size)]

def main():
    args = PARSER.parse_args()
    group_size = args.objects
    prop_size = args.props if args.props > 0 else 0
    marker = args.marker
    num_marker = args.l if args.l is not None and args.l > 0 else 1
    add_props = prop_size > 0
    add_markers = len(marker) > 0 if marker is not None else False
    requests = []

    folder_name = 'results/{}_o_{}_p_{}_m_{}_l_{}'.format(
        str(datetime.now().strftime('%m%d-%H%M')), group_size, prop_size,
        ' '.join(marker)[:20], num_marker)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    LOG = get_logger(folder_name, 'q')
    LOG.info('Group size: {}'.format( group_size))
    LOG.info('Property size: {}'.format(prop_size))
    LOG.info('Markers: {}'.format(marker))
    LOG.info('Minimal markers to match: {}'.format(num_marker))

    with open('../dbpedia/arg/conceptList.txt', encoding='utf-8') as l:
        files = list(l)
        resp_counter = 0
        query_counter = 0
        for line in files:
            try:
                object_name = line.strip().replace(' ', '_')
                objects = file_to_list(
                    '../dbpedia/arg/obj/{}.json'.format(object_name))
                properties = []
                if add_props:
                    properties = file_to_list('../dbpedia/arg/prop/{}.json'.
                                              format(object_name))[:prop_size]
                    LOG.info('Properties: {}'.format(properties))
                shuffle(objects)
                LOG.info(
                    'Start {} (size {})'.format(object_name, len(objects)))
                partitions = partition_to_size(group_size, objects)
                cnt = 0
                for partition in partitions:
                    parameters = [
                        query_string_or(partition, min_match=2)
                    ]
                    if add_markers:
                        parameters.append(query_string_or(marker, min_match=num_marker))
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
                response = grequests.map(requests, size=REQUEST_THREADS, exception_handler=lambda x,y: LOG.fatal(y))
                requests = []
                for result in response:
                    resp_counter += 1
                    if result and result.status_code is 200 and result.json(
                    )['hits']['total'] > 0:
                        file_name = '{}/{}.json'.format(folder_name,
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
                LOG.info(e)
            except Exception as x:
                LOG.info(x)
        LOG.info('Responses {} | (Queries {})'.format(resp_counter, query_counter))


if __name__ == '__main__':
    main()
