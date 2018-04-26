import json

def file_to_list(file):
    """read data file"""
    with open(file, encoding='utf-8') as data:
        array = data.readlines()[0]
        lst = json.loads(array)
        return list(map(lambda entry: entry['label'], lst))


def jsonify(lst, joiner=' OR '):
    quote = ['"{}"'.format((x.encode('utf-8').decode('utf-8'))) for x in lst]
    joined = joiner.join(quote)
    return json.dumps(joined)


def jsonify_no(lst, joiner=' OR '):
    quote = ['"{}"'.format((x.encode('utf-8').decode('utf-8'))) for x in lst]
    joined = joiner.join(quote)
    return json.dumps(joined)[1:-1]


def query_string_or(words, min_match=0):
    quoted = jsonify(words)
    mms = ''
    if min_match > 0:
        mms = '"minimum_should_match" : {},'.format(min_match)
    return '{{ "query_string" : {{ {} "default_field": "text", "query" : {} }} }}'.format(
        mms, quoted)