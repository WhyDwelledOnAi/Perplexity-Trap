import os
import json
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from config import Config
from model import ANCE, TASB, Contriever

args = Config()
retriever = EvaluateRetrieval(DRES(ANCE(), batch_size=args.batch_size), k_values=[1, 3, 5], score_function='dot')

def calc_perplexity(temp):
    path = "" # [TODO]: fill in the path
    with open(path, 'r', encoding='utf8') as f:
        ppls = []
        for line in f:
            data = json.loads(line)
            ppl_list = data['ppl']
            ppl_mean = sum(ppl_list) / len(ppl_list)
            ppls.append(ppl_mean)
    ppls_mean = sum(ppls) / len(ppls)
    return ppls_mean

def calc_rel(temp):
    corpus_path = "" # [TODO]: fill in the path
    corpus, queries, qrels = GenericDataLoader(
            corpus_file=corpus_path, 
            query_file=args.query_path, 
            qrels_file=args.qrels_human_path).load_custom()
    results = retriever.retrieve(corpus, queries)
    
    rels = []
    with open(args.qrels_human_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            qid, did, score = line.strip().split()
            score = int(score)
            if score == 0:
                continue
            if qid not in results.keys():
                continue
            if did not in results[qid].keys():
                continue
            rels.append(results[qid][did])
    rels_mean = sum(rels) / len(rels)
    return rels_mean

if __name__ == '__main__':
    temps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for temp in temps:
        ppl = calc_perplexity(temp)
        rel = calc_rel(temp)
        print(temp, ppl, rel)
    