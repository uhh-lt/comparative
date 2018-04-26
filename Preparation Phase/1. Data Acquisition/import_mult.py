import codecs
import json
import grequests
from datetime import datetime
import linecache
START_INDEX=3118273864

REQUEST_THREADS = 15

FIRE_AFTER = 15
BATCH_SIZE =10000 
INDEX_NAME = 'commoncrawl2'
ES_ENDPOINT = "http://localhost:9222/{}/{}".format(INDEX_NAME, INDEX_NAME)
ID_PREFIX = 'T'
FILE_NAME = 'crawlat'

POOL = grequests.Pool(REQUEST_THREADS)


class IndexBuilder(object):
    def __init__(self):
        self._index = INDEX_NAME
        self._dtype = "sentence"
        self._max_id = START_INDEX
        self._requests = []

    def _get_meta(self):
        meta = { "index": {
            "_index" : self._index,
            "_id": ID_PREFIX+str(self._max_id),
            "_type" : self._dtype}}
        self._max_id += 1
        return json.dumps(meta)

    def _create_document(self, text):
        doc = {"text": text.strip()}
        return json.dumps(doc)

    def e_handler(self,req,exception):
        print(req)
        print(exception)


    def create_index(self, raw_sentences_fpath, output_fpath):
        num = 0
        req_nmbr = 0
        output_f = codecs.open(output_fpath, "w", "utf-8")
        with codecs.open(raw_sentences_fpath, "r", "utf-8") as input_f:

            for i, line in enumerate(input_f):
                if True :

                    if i % BATCH_SIZE == 0 and i != 0:
                        print('{} Request'.format(i))
                        output_f.close()
                        self.create_request(output_fpath)
                        output_f = codecs.open(output_fpath, "w", "utf-8")
                        num = 0

                    if len(self._requests) == FIRE_AFTER:
                      #  resp = grequests.imap(self._requests, exception_handler=self.e_handler)
                       # for response in resp:
                       #     print(response.status_code)                        
                        for i, req in enumerate(self._requests):
                            print('{} Request {}'.format(
                                req_nmbr, str(datetime.now().strftime(
                                    '%H:%M'))))
                            req_nmbr = req_nmbr+1
                            grequests.send(req, POOL)
                       #     print(resp)
                        self._requests = []
                    print(self._get_meta(), file=output_f)
                    print(self._create_document(line), file=output_f)
                    num += 1
                elif i % (BATCH_SIZE*10) == 0:
                    print('Skipped {}/{} ({:2.2f})'.format(i,START_INDEX,(i/START_INDEX)))
        print("*")
        self.create_request(output_fpath)
        output_f.close()



    def create_request(self, index_fpath):
        with open(index_fpath, 'rb') as input_f:
            data = input_f.read()
            res = grequests.post(
                url=ES_ENDPOINT + "/_bulk",
                data=data,
                headers={'Content-Type': 'application/x-ndjson'})
            self._requests.append(res)



#usages_fpath = "/home/panchenko/tmp/usage/usages-wiki.csv"
#pcz_fpath = "/home/panchenko/tmp/ddt-mwe-45g-8m-thr-agressive2-cw-e0-N200-n200-minsize5-isas-cmb-313k-hyper-filter-closure.csv.gz"
raw_sentences_fpath = "/srv/data/arguments/{}".format(FILE_NAME)
output_fpath = raw_sentences_fpath + ".index.json"
ib = IndexBuilder()
ib.create_index(raw_sentences_fpath, output_fpath)
#find_examples(pcz_fpath, usages_fpath)

