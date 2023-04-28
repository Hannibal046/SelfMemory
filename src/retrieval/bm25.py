import pickle
import json
import argparse
import os
import logging
import json
import tqdm
from elasticsearch import Elasticsearch
import re
import sacrebleu
import editdistance

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--build_index', action='store_true',
        help='whether to train an index from scratch')
    parser.add_argument('--search_index', action='store_true',
        help='whether to search from a built index')
    parser.add_argument('--index_file', type=str)
    parser.add_argument('--search_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--query_lang', type=str)
    # parser.add_argument('--response_lang', type=str)
    parser.add_argument('--start_index', type=int,default=0)
    parser.add_argument('--end_index', type=int,default=-1)
    parser.add_argument('--index_name', type=str,default='used_once')
    parser.add_argument('--topk', type=int, default=10)
    # parser.add_argument('--allow_hit', action='store_true')
    return parser.parse_args()

def debpe(bpe):
    return re.sub(r'(@@ )|(@@ ?$)', '', bpe)

def get_unedited_words(src_sent, src_tm_sent):
    # Here first sentence is the edited sentence and the second sentence is the target sentence
    # If src_sent is the first sentence, then insertion and deletion should be reversed.
    """ Dynamic Programming Version of edit distance and finding unedited words."""
    a = src_sent.split()
    b = src_tm_sent.split()
    edit_distance = editdistance.eval(a, b)

    edit_distance = 1 - edit_distance / max(len(src_sent), len(src_tm_sent))

    return edit_distance

def get_topk_sent_id(src, src_sim, k=6):
    scores = list(map(lambda x: -get_unedited_words(src, x), src_sim))
    topk = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])[:k]
    ans = [it[1] for it in topk]
    return ans

def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    es = Elasticsearch([{u'host': "localhost", u'port': "9200"}])
    if args.build_index:
        # queries, responses = [], []
        queries = []
        responses  = []
        with open(args.index_file,'r') as f:
            for idx,line in enumerate(f.readlines()):
                line = json.loads(line)
                queries.append(" ".join(line[args.query_lang].split()[:600]))
                responses.append(idx)
                # responses.append(line[args.response_lang])
        # with open(args.index_file, 'r') as f:
        #     for line in f.readlines():
        #         q, r = line.strip().split('\t')
        #         queries.append(q)
        #         responses.append(r)

        logger.info('build with elasticsearch')
        from elasticsearch.helpers import bulk
        body = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": "standard"
                    },
                    "number_of_shards": "1",
                    "number_of_replicas": "1",
                }
            },
            "mappings": {
                "properties": {
                    "query": {
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "standard",
                    },
                    "response": {
                        "type": "text",
                    }
                }
            }
        }
        if es.indices.exists(index=args.index_name):
            es.indices.delete(index=args.index_name)
        es.indices.create(index=args.index_name, body=body)

        index = args.index_name

        actions = []

        for idx, (query, response) in tqdm.tqdm(enumerate(zip(queries, responses)), total=len(queries)):
            action = {
                "_index": index,
                "_source": {
                    # "query": debpe(query),
                    "query": query,
                    "response": response
                }
            }
            actions.append(action)
            if len(actions) >= 1000:
                success, _ = bulk(es, actions, raise_on_error=True)
                actions = []
        if actions:
            success, _ = bulk(es, actions, raise_on_error=True)
        es.indices.refresh(index)
        info = es.indices.stats(index=index)
        print('total document indexed', info["indices"][index]["primaries"]["docs"]["count"]) 

    if args.search_index:
        queries = []
        print(args.search_file,args.start_index,args.end_index)

        with open(args.search_file, 'r') as f:
            if args.end_index > 0:
                _t = []
                idx = 0
                for x in f:
                    if idx >= args.start_index and idx < args.end_index:
                        _t.append(x)
                    idx += 1
                # _t = f.readlines()[args.start_index:args.end_index]
            else:
                _t = f.readlines()
            for line in _t:
            # for line in f.readlines():
                line = json.loads(line)
                temp_query = line[args.query_lang]
                temp_query = " ".join(temp_query.split()[:600])
                # q, r = line.strip().split('\t')
                queries.append(temp_query)
                # responses.append()

        logger.info('search with elasticsearch')

        index = args.index_name
        query_body = {
                "query": {
                "match":{
                     "query": None
                 }
                },
                "size": args.topk
        }

        ret = []
        miss_cnt = 0
        # with open(args.output_file,'w') as fo:
        # for idx,query in enumerate(tqdm.tqdm(queries, total=len(queries))):
        for idx,query in enumerate(tqdm.tqdm(queries)):
            # query_body["query"]["match"]["query"] = debpe(query)
            query_body["query"]["match"]["query"] = query
            es_result = es.search(index=index, body=query_body)
            ret_q = [item["_source"]["query"] for item in es_result["hits"]["hits"]]
            ret_r = [item["_source"]["response"] for item in es_result["hits"]["hits"]]
            
            
            if len(ret_q) == 0 or len(ret_r) == 0:
                ret.append([])
                miss_cnt += 1
                continue
            ret.append(ret_r)
        pickle.dump(ret,open(args.output_file,'wb'))
        print(miss_cnt)

if __name__ == "__main__":
    args = parse_args()
    main(args)
