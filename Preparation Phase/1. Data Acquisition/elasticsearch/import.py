import codecs
import json
import requests
from requests.auth import HTTPBasicAuth

ES_ENDPOINT = "http://localhost:9222/commoncrawl/commoncrawl"

class IndexBuilder(object):
    def __init__(self):
        self._index = "commoncrawl"
        self._dtype = "sentence"
        self._max_id = 0

    def _get_meta(self):
        meta = { "index": {
            "_index" : self._index,
            "_id": self._max_id,
            "_type" : self._dtype}}
        self._max_id += 1
        return json.dumps(meta)

    def _create_document(self, text):
        doc = {"text": text.strip()}
        return json.dumps(doc)

    def create_index(self, raw_sentences_fpath, output_fpath):
        num = 0
        output_f = codecs.open(output_fpath, "w", "utf-8")
        with codecs.open(raw_sentences_fpath, "r", "utf-8") as input_f:
            for i, line in enumerate(input_f):
                if i % 100000 == 0 and i != 0:
                    print "*"
                    output_f.close()
                    self.index_file(output_fpath)
                    output_f = codecs.open(output_fpath, "w", "utf-8")
                    num = 0
                print >> output_f, self._get_meta()
                print >> output_f, self._create_document(line)
                num += 1

        print "*"
        self.index_file(output_fpath)
        output_f.close()

    def index_file(self, index_fpath):

        with open(index_fpath, 'rb') as input_f:
            data = input_f.read()
            res = requests.post(
                url=ES_ENDPOINT + "/_bulk",
                data=data,
                headers={'Content-Type': 'application/x-ndjson'})

            print res
            print res.content[:80]


"""from jnt.pcz.sense_clusters import SenseClusters


STOPWORDS = ["thing"]
ES_ENDPOINT = "http://localhost:9200"
MAX_USAGE_EXAMPLES = 5


def get_usage_examples(result, target):
    usage_examples = []
    for j, hit in enumerate(result["hits"]["hits"]):
        if j >= MAX_USAGE_EXAMPLES: break
        if hit["_source"]["text"].strip() == target: continue
        usage_examples.append(hit["_source"]["text"])
    return usage_examples

def make_es_query(lucene_query):
    es_query = {
        "query": {
            "query_string" : {
                "default_field" : "text",
                "query" : lucene_query
            }
        }
    }    
    return json.dumps(es_query)

def es_search(lucene_query):
    response = requests.post(
        ES_ENDPOINT + "/wsd/_search",
        data=make_es_query(lucene_query),
        headers={'Content-Type': 'application/json'})
    
    return response

def make_or_list(word_list):
    words = [word_sense.split("#")[0].replace('"','') for word_sense in word_list]
    filtered_words = [word for word in words if word not in STOPWORDS]
    if len(filtered_words) > 1:
        return " OR ".join('"%s"' % w for w in filtered_words)
    else:
        return '""'
    
def make_query(target, hypernyms, related, query_type):
    if query_type == "strong":
        query = '"%s is a" AND (%s) AND (%s)' % (
            target,
            hypernyms,
            related)
    elif query_type == "medium":
        query = '"%s" AND (%s) AND (%s)' % (
            target,
            hypernyms,
            related)
    else: # query_type == "weak":
        query = '"%s" AND (%s OR %s)' % (
            target,
            hypernyms,
            related)
        
    #print ">>>>", query, "\n"
        
    return query

def get_usages_from_query(target, hypernyms, related, query_type):
    query = make_query(target, hypernyms, related, query_type)
    response = es_search(query)
    
    if response.status_code == 200:
        result = json.loads(response.content)    
        if result["hits"]["total"] > 0:
            return get_usage_examples(result, target)
        
    return []

def find_examples(pcz_fpath, usages_fpath):
    pcz = SenseClusters(pcz_fpath,
                        strip_dst_senses=False,
                        load_sim=True,
                        verbose=False,
                        normalized_bow=False,
                        use_pickle=True,
                        voc_fpath="",
                        voc=[],
                        normalize_sim=False)

    with codecs.open(usages_fpath, "w", "utf-8") as out:
        sentence_id = 0
        print >> out, "sentence_id\tsense_id\tsense_position\tsentence"
        for i, target in enumerate(pcz.data):
            senses = pcz.get_senses_full(target)
            for sense_id in senses:

                hypernyms = make_or_list(senses[sense_id]["isas"])
                related = make_or_list(senses[sense_id]["cluster"])

                usages = get_usages_from_query(target, hypernyms, related, "strong")
                if len(usages) < MAX_USAGE_EXAMPLES:
                    usages += get_usages_from_query(target, hypernyms, related, "medium")
                    if len(usages) < MAX_USAGE_EXAMPLES:
                        usages = get_usages_from_query(target, hypernyms, related, "weak")

                if len(usages) > 0:
                    usages = usages[:MAX_USAGE_EXAMPLES]
                    for sentence in usages:
                        beg = sentence.lower().find(target.lower())
                        end = beg + len(target) if beg != -1 else -1
                        sense_position = "%d,%d" % (beg, end)
                        print >> out, "%s\t%s#%s\t%s\t%s" % (sentence_id, target, sense_id, sense_position, sentence)
                        sentence_id += 1
                else:
                    print >> out, "%s\t%s#%s\t%s" % (sentence_id, target, sense_id, sense_position)
                    sentence_id += 1

                print i, target, sense_id, "=====", len(usages)

   """
#usages_fpath = "/home/panchenko/tmp/usage/usages-wiki.csv"
#pcz_fpath = "/home/panchenko/tmp/ddt-mwe-45g-8m-thr-agressive2-cw-e0-N200-n200-minsize5-isas-cmb-313k-hyper-filter-closure.csv.gz"
raw_sentences_fpath = "/srv/data/arguments/cc-2016-en-nohtml-nonoise-sort.txt"
output_fpath = raw_sentences_fpath + ".index.json"
ib = IndexBuilder()
ib.create_index(raw_sentences_fpath, output_fpath)
#find_examples(pcz_fpath, usages_fpath)