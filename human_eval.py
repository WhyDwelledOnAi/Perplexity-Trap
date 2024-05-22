import json
import numpy as np
import pandas as pd


from config import Config

args = Config()

def load_pairids(dataset):
    qrels_path = "{}data/cocktail/{}/qrels/human.tsv".format(args.root_dir, dataset)

    pair_ids = list()
    with open(qrels_path, 'r', encoding='utf8') as f:
        for line in f:
            data = line.strip().split('\t')
            if data[2] == 'score' or int(data[2]) == 0:
                continue
            pair_ids.append((data[0], data[1]))
    return pair_ids

def sample_document(dataset, sample_num):
    pair_ids = load_pairids(dataset)
    doc_ids = set([did for (qid, did) in pair_ids])

    human_path = f"{args.root_dir}data/cocktail/{dataset}/corpus/human.jsonl"
    id2doc = {"human": {}}
    with open(human_path, 'r', encoding='utf8') as f:
        for line in f:
            info = json.loads(line)
            if info['_id'] not in doc_ids:
                continue
            id2doc["human"][info['_id']] = info["text"]
    query_path = f"{args.root_dir}data/cocktail/{dataset}/queries/queries.jsonl"
    id2query = {}
    with open(query_path, 'r', encoding='utf8') as f:
        for line in f:
            info = json.loads(line)
            id2query[info['_id']] = info["text"]
    
    eval_samples = []
    for temp in ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]:
        id2doc[temp] = {}
        path = f"{args.root_dir}data/cocktail/{dataset}/corpus/tmp{temp}-ori.jsonl"
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                info = json.loads(line)
                id2doc[temp][info['_id']] = info["text"]
        
        sample_index = np.random.choice([i for i in range(len(pair_ids))], sample_num, replace=False)
        doc_ids_sample = [pair_ids[i] for i in sample_index]
        doc1_is_human = np.random.choice([0, 1], sample_num, replace=True)
        for i in range(sample_num):
            info = {"doc_id": doc_ids_sample[i], 
                    "temp": temp,
                    "doc1_is_human": doc1_is_human[i],
                    "query": id2query[doc_ids_sample[i][0]],
                    "doc1": id2doc["human"][doc_ids_sample[i][1]] if doc1_is_human[i] == 1 else id2doc[temp][doc_ids_sample[i][1]],
                    "doc2": id2doc["human"][doc_ids_sample[i][1]] if doc1_is_human[i] == 0 else id2doc[temp][doc_ids_sample[i][1]],
                    "label1": None,"label2": None,"label3": None,"label": None}
            eval_samples.append(info)
        
    return eval_samples

def save_excel(sample_num):
    datasets = ["dl19", "scidocs", "trec-covid"]
    writer = pd.ExcelWriter("./human_eval.xlsx")
    for dataset in datasets:
        eval_samples = sample_document(dataset, sample_num)
        df = pd.DataFrame(eval_samples)
        df.to_excel(writer, sheet_name=dataset, index=False)
    writer._save()
    writer.close()

def voting(labels):
    if labels.count('C') >= 2:
        return 'C'
    elif labels.count('B') >= 2:
        return 'B'
    elif labels.count('A') >= 2:
        return 'A'
    else:
        raise ValueError("No majority voting")
    

def read_excel(path='./human_eval.xlsx'):
    df = pd.read_excel(path, sheet_name='dl19')
    
    labels_list = {0.0: [], 0.2: [], 0.4: [], 0.6: [], 0.8: [], 1.0: []}
    consistency_list = {0.0: {"Human": 0, "LLM": 0, "Equal": 0}, 
                        0.2: {"Human": 0, "LLM": 0, "Equal": 0},
                        0.4: {"Human": 0, "LLM": 0, "Equal": 0},
                        0.6: {"Human": 0, "LLM": 0, "Equal": 0},
                        0.8: {"Human": 0, "LLM": 0, "Equal": 0},
                        1.0: {"Human": 0, "LLM": 0, "Equal": 0}}
    
    for i in range(len(df)):
        evals = [df.iloc[i]["label1"], df.iloc[i]["label2"], df.iloc[i]["label3"]]
        vote_result = voting(evals)

        if vote_result == 'C':
            label = "Equal"
        elif vote_result == 'B':
            label = "Doc2"
        elif vote_result == 'A':
            label = "Doc1"
        
        if df.iloc[i]["doc1_is_human"] == 0:
            if label == "Doc1":
                label = "LLM"
            elif label == "Doc2":
                label = "Human"
        else:
            if label == "Doc2":
                label = "LLM"
            elif label == "Doc1":
                label = "Human"

        labels_list[df.iloc[i][ "temp"]].append(label)
        if evals.count(vote_result) == 3:
            consistency_list[df.iloc[i][ "temp"]][label] += 1
    
    # print(labels_list.count("Human"), labels_list.count("LLM"), labels_list.count("Equal"))
    for temp in labels_list:
        print(temp)
        print(labels_list[temp].count("Human") / len(labels_list[temp]), labels_list[temp].count("LLM") / len(labels_list[temp]), labels_list[temp].count("Equal") / len(labels_list[temp]))
        print(consistency_list[temp]["Human"] / (labels_list[temp].count("Human") + 0.1), 
              consistency_list[temp]["LLM"] / (labels_list[temp].count("LLM") + 0.1), 
              consistency_list[temp]["Equal"] / (labels_list[temp].count("Equal") + 0.1))
    


if __name__ == '__main__':
    save_excel(20)
    read_excel()