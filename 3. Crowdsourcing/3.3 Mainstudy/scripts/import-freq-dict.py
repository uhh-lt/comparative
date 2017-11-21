import requests
import codecs
import spacy

ES_ENDPOINT = "http://localhost:9222/freq-dict/freq-dict"

nlp = spacy.load('en')

with open('freq-59g-mwe62m.csv', encoding='latin-1') as data:
#with open('foo.csv', encoding='latin-1') as data:
    pl = ''
    for i, line in enumerate(list(data.readlines())):
        if i > 0:
            word, freq = line.split('\t')
            json = """{{ "index": {{"_index" : "freq-dict", "_type":"freq", "_id":"{}" }}}}""".format(
                i)
            payload = '{{ "freq" : "{}", "word" : "{}", "lemma" : "{}", "entity" : "{}"}}'.format(
                freq.rstrip().encode('utf-8').decode('utf-8'),
                word.rstrip().encode('utf-8').decode('utf-8'), [
                    token.lemma_
                    for token in nlp(
                        word.rstrip().encode('utf-8').decode('utf-8'))
                ], [
                    token.ent_type_
                    for token in nlp(
                        word.rstrip().encode('utf-8').decode('utf-8'))
                ])
            if i % 1000 != 0:
                pl += json + '\n' + payload +'\n'
            else:
                pl += '\n'
                res = requests.post(
                    url=ES_ENDPOINT + "/_bulk",
                    data=pl)
                res.raise_for_status()
                print(i, res)
                pl = ''
