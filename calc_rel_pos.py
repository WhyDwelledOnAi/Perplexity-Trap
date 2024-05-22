import json
import pickle
from tqdm import tqdm

from config import Config 
from model import *

args = Config()


def load_pos_pair(dataset):
    qrels_path = "{}data/cocktail/{}/qrels/human.tsv".format(args.root_dir, dataset)
    doc_ids, query_ids = set(), set()

    with open(qrels_path, 'r', encoding='utf8') as f:
        for line in f:
            data = line.strip().split('\t')
            doc_ids.add(data[1])
            doc_ids.add("-" + data[1])
            query_ids.add(data[0])
    
    
    id2text_doc, id2text_query = {}, {}
    corpus_path = "{}data/cocktail/{}/corpus/merge.jsonl".format(args.root_dir, dataset)
    with open(corpus_path, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            if data['_id'] in doc_ids:
                id2text_doc[data['_id']] = data['text']
    query_path = "{}data/cocktail/{}/queries/queries.jsonl".format(args.root_dir, dataset)
    with open(query_path, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            if data['_id'] in query_ids:
                id2text_query[data['_id']] = data['text']

    
    pair_list = []
    with open(qrels_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(tqdm(f)):
            if i == 0:
                continue
            data = line.strip().split('\t')
            query_id, doc_id, score = data[0], data[1], data[2]
            if int(score) == 0:
                continue
            corpus = {}
            corpus["human"] = {"text": id2text_doc[doc_id], "title":""}
            corpus["llm"] = {"text": id2text_doc["-" + doc_id], "title":""}
            query = {}
            query["query"] = id2text_query[query_id]
            query["abcde"] = id2text_query[query_id]
            pair_list.append((query_id, doc_id, query, corpus))
            
    return pair_list

def save_pos_pair(dataset, pair_list):
    save_path = "{}data/cocktail/{}/pos_pair.jsonl".format(args.root_dir, dataset)
    with open(save_path, 'w', encoding='utf8') as f:
        for pair in tqdm(pair_list):
            f.write(json.dumps(pair) + '\n')


def calc_rel_pos(retriever_name, dataset, pair_list):
    results_path = "{}data/cocktail/{}/pos_rel_{}.jsonl".format(args.root_dir, dataset, retriever_name)

    retriever = load_retriever(retriever_name, k_values=[2])
    for query_id, doc_id, query, corpus in tqdm(pair_list):
        result = retriever.retrieve(corpus, query)['query']
        human_score = result["human"]
        llm_score = result["llm"]
        with open(results_path, 'a+', encoding='utf8') as f:
            f.write(json.dumps({"query_id": query_id, "doc_id": doc_id, "human": human_score, "llm": llm_score}) + '\n')
    

def normalize(human, llm):
    conbine = human + llm
    mean = sum(conbine) / len(conbine)
    std = (sum([(i - mean) ** 2 for i in conbine]) / len(conbine)) ** 0.5
    human = [(i - mean) / std for i in human]
    llm = [(i - mean) / std for i in llm]
    return human, llm


def calc_mean(retriever, dataset):
    rel_pos_path = "{}data/cocktail/{}/pos_rel/{}.pkl".format(args.root_dir, dataset, retriever)
    with open(rel_pos_path, 'rb') as f:
        results = pickle.load(f)
    rel_human_list = []
    rel_llm_list = []
    for qid, dtuple in results.items():
        for did, score in dtuple.items():
            if did.startswith("-"):
                rel_llm_list.append(score)
            else:
                rel_human_list.append(score)
    rel_human_list, rel_llm_list = normalize(rel_human_list, rel_llm_list)
    print("rel_human: {}, rel_llm: {}".format(np.mean(rel_human_list), np.mean(rel_llm_list)))

if __name__ == '__main__':
    for retriever_name in ['bert', 'roberta', 'ance', 'tasb', 'contriever', 'cocondenser']:
        for dataset in ['trec-covid', 'scidocs', 'dl19']:
            print(retriever_name, dataset)
            pair_list = load_pos_pair(dataset)
            calc_rel_pos(retriever_name, dataset, pair_list)
