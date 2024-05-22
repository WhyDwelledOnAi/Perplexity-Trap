import json
import pandas as pd
from scipy import stats
import numpy as np
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from config import Config
from model import load_data, merge_dict, load_retriever, calc_delta_metric
from model import load_perplexity as load_ppl

args = Config()

def load_dids(dataset, sample_num=128):
    qrels_path = "{}data/cocktail/{}/qrels/human.tsv".format(args.root_dir, dataset)

    doc_ids = list()
    with open(qrels_path, 'r', encoding='utf8') as f:
        for line in f:
            data = line.strip().split('\t')
            if data[2] == 'score' or int(data[2]) == 0:
                continue
            doc_ids.append(data[1])
    if sample_num != -1:
        doc_ids = np.random.choice(doc_ids, sample_num, replace=False)
        doc_ids = doc_ids.tolist()
    
    doc_ids_set = set()
    for did in doc_ids:
        doc_ids_set.add(did)
        doc_ids_set.add('-'+did)
    return doc_ids_set

def load_perplexity(dataset, doc_ids):
    ppl_path = "{}data/cocktail/{}/perplexity.jsonl".format(args.root_dir, dataset)
    source_list = []
    perplexity_list = []
    with open(ppl_path, 'r', encoding='utf8') as f:
        for line in f:
            info = json.loads(line)
            did = info['_id']
            if did not in doc_ids:
                continue
            ppl_list = info['ppl']
            ppl_mean = sum(ppl_list) / len(ppl_list)
            source = 1. if did.startswith('-') else 0.
            source_list.append(source)
            perplexity_list.append(ppl_mean)

    return source_list, perplexity_list

def load_relevancy(retriever_name, dataset, ppl_human, ppl_llm, doc_ids):
    rel_dir =  "{}data/cocktail/{}/pos_rel_{}.jsonl".format(args.root_dir, dataset, retriever_name)
    ppl_list = []
    rel_list = []
    with open(rel_dir, 'r', encoding='utf8') as f:
        for line in f:
            info = json.loads(line)
            if info["doc_id"] not in doc_ids:
                continue
            ppl_list.append(ppl_human)
            rel_list.append(info['human'])
            ppl_list.append(ppl_llm)
            rel_list.append(info['llm'])
    return ppl_list, rel_list

def first_stage_reg(source_list, perplexity_list):
    df = pd.DataFrame({'source': source_list, 'ppl': perplexity_list})
    generated_df = df[df['source'] == 1]
    human_df = df[df['source'] == 0]
    generated_ppl_mean = generated_df['ppl'].mean()
    human_ppl_mean = human_df['ppl'].mean()
    return generated_ppl_mean, human_ppl_mean

def second_stage_reg(ppl_list, rel_list):
    slope, intercept, r_value, p_value, std_err = stats.linregress(ppl_list, rel_list)
    return slope, p_value


def calibration(retriever_name, dataset_name, results, did2perplexity, debias_coef):
    coef = debias_coef[retriever_name][dataset_name]
    
    results_new = copy.deepcopy(results)
    for query_id, doc_rels in results.items():
        for doc_id, rel in doc_rels.items():
            if doc_id not in did2perplexity:
                raise ValueError("doc_id <{}> not in did2perplexity.".format(doc_id))
            
            amendment = coef * did2perplexity[doc_id]
            results_new[query_id][doc_id] = rel - amendment
    return results_new

def check_calibration(retriever_name, dataset, debias_coef):
    corpus, queries, qrels_human, qrels_llm = load_data(dataset)
    qrels = merge_dict(qrels_human, qrels_llm)

    retriever = load_retriever(retriever_name, k_values=[1, 3, 5])

    rel_dir = "{}data/cocktail/{}/relevant_score/{}.jsonl".format(args.root_dir, dataset, retriever_name)
    results = dict()
    with open(rel_dir, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            info = json.loads(line)
            results.update({info['qid']: info['rel']})
            

    # print("begin calibration...")
    results = calibration(retriever_name, dataset, results, load_ppl(dataset), debias_coef)
    
    ndcg, map_, _, __ = retriever.evaluate(qrels, results, retriever.k_values)
    # print("ndcg@3:", ndcg['NDCG@3'])

    ndcg_human, map_human, _, __ = retriever.evaluate(qrels_human, results, retriever.k_values)
    ndcg_llm, map_llm, _, __ = retriever.evaluate(qrels_llm, results, retriever.k_values)
    delta_ndcg = calc_delta_metric(ndcg_human, ndcg_llm)
    delta_map = calc_delta_metric(map_human, map_llm)
    # print("delta_ndcg@3:", delta_ndcg['NDCG@3'])
    return ndcg['NDCG@3'], delta_ndcg['NDCG@3']


if __name__ == '__main__':
    for sample_num in [64, 128, 256, -1]:
        for sample_time in range(5):                
            debias_coef = {'bert':  {'scidocs': 0.0, 'trec-covid': 0.0, 'dl19': 0.0},
                        'roberta':  {'scidocs': 0.0, 'trec-covid': 0.0, 'dl19': 0.0},
                        'ance':  {'scidocs': 0.0, 'trec-covid': 0.0, 'dl19': 0.0},
                        'tasb':  {'scidocs': 0.0, 'trec-covid': 0.0, 'dl19': 0.0},
                    'contriever':  {'scidocs': 0.0, 'trec-covid': 0.0, 'dl19': 0.0},
                'cocondenser':  {'scidocs': 0.0, 'trec-covid': 0.0, 'dl19': 0.0}}
            
            doc_ids = load_dids('dl19', sample_num=sample_num)
            source_list, perplexity_list = load_perplexity('dl19', doc_ids)
            ppl_llm, ppl_human = first_stage_reg(source_list, perplexity_list)
            for retriever in ['bert', 'roberta', 'ance', 'tasb', 'contriever', 'cocondenser']:
                ppl_list, rel_list = load_relevancy(retriever, 'dl19', ppl_human, ppl_llm, doc_ids)
                slope, p_value = second_stage_reg(ppl_list, rel_list)
                debias_coef[retriever]['scidocs'] = slope
                debias_coef[retriever]['trec-covid'] = slope
                debias_coef[retriever]['dl19'] = slope
                
                for dataset in ['scidocs', 'trec-covid', 'dl19']:
                    ndcg, delta_ndcg = check_calibration(retriever, dataset, debias_coef)
                    info = {"sample_num": sample_num, 
                            "sample_time": sample_time, 
                            "retriever": retriever,
                            "slope": slope,  
                            "dataset": dataset,
                            "ndcg": ndcg, 
                            "delta_ndcg": delta_ndcg}
                    with open("./calibration_result.jsonl", 'a+', encoding='utf8') as f:
                        f.write(json.dumps(info) + "\n")

            
    