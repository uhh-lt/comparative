import json

t = ['test', 'a b c']

def jsonify(lst):
    quote = ['"{}"'.format(x) for x in lst ]
    joined =  ' OR '.join(quote)
    return json.dumps(joined)
    
print(jsonify(t))