from beir.datasets.data_loader import GenericDataLoader


from config import Config
from model import *

args = Config()


def load_data(dataset):
    query_path = "{}data/cocktail/{}/queries/queries.jsonl".format(args.root_dir, dataset)
    qrels_human_path = "{}data/cocktail/{}/qrels/human.tsv".format(args.root_dir, dataset)
    corpus_merge_path = "{}data/cocktail/{}/corpus/merge.jsonl".format(args.root_dir, dataset)
    
    corpus, queries, qrels_human = GenericDataLoader(
            corpus_file=corpus_merge_path, 
            query_file=query_path, 
            qrels_file=qrels_human_path).load_custom()
    
    
    return corpus, queries, qrels_human

def save_results(retriever_name, dataset, results):
    save_path = "{}data/cocktail/{}/relevant_score/{}.jsonl".format(args.root_dir, dataset, retriever_name)
    with open(save_path, 'w', encoding='utf8') as f:
        for qid, rel_scores in results.items():
            info = {"qid": qid, "rel": rel_scores}
            f.write(json.dumps(info) + '\n')

def calc_rel(retriever_name, dataset, k_values=[1, 3, 5]):
    corpus, queries, qrels_human = load_data(dataset)
    retriever = load_retriever(retriever_name, k_values=k_values)
    
    results = retriever.retrieve(corpus, queries)
    save_results(retriever_name, dataset, results)
    
    del corpus, queries, qrels_human
    del retriever
    del results

if __name__ == '__main__':
    calc_rel(args.retriever, args.dataset)