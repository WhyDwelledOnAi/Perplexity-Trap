from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm import tqdm, trange
import torch
from transformers import BertForMaskedLM, AutoTokenizer

from nltk.tokenize import word_tokenize
from nltk import pos_tag
import random
import os
import copy
import pickle
import json
from scipy.stats import pearsonr




from config import Config
from model import *


args = Config()

tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
model = BertForMaskedLM.from_pretrained(args.bert_dir).to(args.gpu)
retriever = None
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")


def load_corpus():
    print("Loading Corpus...")
    corpus= dict()
    with open(args.corpus_merge_path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            id = data['_id']
            text = data['text']
            tokens = word_tokenize(text)
            text = " ".join(tokens)
            corpus[id] = text
    return corpus
def load_queries():
    print("Loading queries...")
    queries = dict()
    with open(args.query_path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            id = data['_id']
            text = data['text']
            queries[id] = text
    return queries
def load_qrels():
    print("Loading qrels...")
    qrels = list()
    with open(args.qrels_human_path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            if line.startswith("query-id"):
                continue
            data = line.strip().split("\t")
            qid = data[0]
            did = data[1]
            score = int(data[2])
            if score == 0:
                continue
            qrels.append((qid, did))
    return qrels
def conbine_data(corpus, queries, qrels):
    data = list()
    for qid, did in qrels:
        info = {
            "qid":qid,
            "did":did,
            "query": queries[qid],
            "doc_human": corpus[did],
            "doc_llm": corpus["-" + did]
        }
        data.append(info)
    return data
def load_synonym():
    print("Loading synonym dictionary...")
    with open(args.synonym_path, 'r', encoding='utf8') as f:
        return json.loads(f.read())

def isMeaningful(tag):
    return tag.startswith("N") or tag.startswith("V") \
        or tag.startswith("J") or tag.startswith("R")
def replace_synonym_one(data, synonym):
    print("Replacing synonym...")

    have_synonym = set(synonym.keys()) 

    data_new = list()
    for info in tqdm(data):
        doc_human = info['doc_human']
        words = copy.deepcopy(doc_human.split(" "))
        tags = pos_tag(words)

        flag = False
        for i, word in enumerate(words):
            if not isMeaningful(tags[i][1]): 
                continue
            if word not in have_synonym: 
                continue
            if tags[i][1][0] not in synonym[word].keys():
                continue
            
            synonyms = synonym[word][tags[i][1][0]]
            words[i] = synonyms[0]
            flag = True
            

        if not flag:
            continue
        text_replace = " ".join(words)
        info['doc_human_replace'] = text_replace
        

        doc_llm = info['doc_llm']
        words = copy.deepcopy(doc_llm.split(" "))
        tags = pos_tag(words)

        flag = False
        for i, word in enumerate(words):
            if not isMeaningful(tags[i][1]):
                continue
            if word not in have_synonym:
                continue
            if tags[i][1][0] not in synonym[word].keys(): 
                continue
            
            synonyms = synonym[word][tags[i][1][0]] 
            words[i] = synonyms[0] 
            flag = True
            

        if not flag: 
            continue
        text_replace = " ".join(words)
        info['doc_llm_replace'] = text_replace
        
        data_new.append(info)


    return data_new


def concat_tokens(tokens, ppls):
    new_tokens, new_ppls = [], []
    cnt = 1
    for i, token in enumerate(tokens):
        if token.startswith('##'):
            new_tokens[-1] += token[2:]
            new_ppls[-1] = new_ppls[-1] + ppls[i]
            cnt += 1
        else:
            if i != 0:
                new_ppls[-1] /= cnt
            cnt = 1
            new_tokens.append(token)
            new_ppls.append(copy.deepcopy(ppls[i]))
    
    return new_tokens, new_ppls
def calc_ppl(text):
    encoded_input = tokenizer([text], return_tensors='pt', 
                              truncation=True, max_length=512, 
                              add_special_tokens=False)
    tokens = [tokenizer.decode(encode_token) for encode_token in encoded_input['input_ids'][0]]
    true_input_ids = encoded_input['input_ids'].squeeze(dim=0).to(args.gpu)

    input_ids = encoded_input['input_ids']
    input_ids = input_ids.repeat(input_ids.shape[1], 1)
    for i in range(input_ids.shape[0]):
        input_ids[i, i] = tokenizer.mask_token_id
    token_type_ids = encoded_input['token_type_ids']
    token_type_ids = token_type_ids.repeat(token_type_ids.shape[1], 1)
    attention_mask = encoded_input['attention_mask']
    attention_mask = attention_mask.repeat(attention_mask.shape[1], 1)

    
    loss = []
    for i in range(0, input_ids.shape[0], args.batch_size):
        encoded_input = {'input_ids': input_ids[i:i+args.batch_size].to(args.gpu), 
                         'token_type_ids': token_type_ids[i:i+args.batch_size].to(args.gpu), 
                         'attention_mask': attention_mask[i:i+args.batch_size].to(args.gpu)}
        with torch.no_grad():
            model_output = model(**encoded_input)[0]
            pred_token_ids = [model_output[j][i+j] for j in range(model_output.shape[0])]
            pred_token_ids = torch.stack(pred_token_ids).to(args.gpu)
            loss += loss_fct(pred_token_ids, true_input_ids[i:i+args.batch_size]).cpu().tolist()
    loss = [round(l, 6) for l in loss]
    # tokens, loss = concat_tokens(tokens, loss)
    return tokens, loss

def calc_ppl_all(data):
    print("Calculating Perplexity...")
    for info in tqdm(data):
        doc_text = info['doc_human']
        tokens, ppls = calc_ppl(doc_text)
        ppl_mean = sum(ppls) / len(ppls)
        info['ppl_human'] = ppl_mean

        doc_text_replace = info['doc_llm']
        tokens, ppls = calc_ppl(doc_text_replace)
        ppl_mean = sum(ppls) / len(ppls)
        info['ppl_llm'] = ppl_mean

        doc_text = info['doc_human_replace']
        tokens, ppls = calc_ppl(doc_text)
        ppl_mean = sum(ppls) / len(ppls)
        info['ppl_human_replace'] = ppl_mean

        doc_text_replace = info['doc_llm_replace']
        tokens, ppls = calc_ppl(doc_text_replace)
        ppl_mean = sum(ppls) / len(ppls)
        info['ppl_llm_replace'] = ppl_mean
    return data


def calc_rel(data, retriever_name):
    global retriever
    retriever = load_retriever(retriever_name, k_values=[4])

    print("Calculating Relevance of {}...".format(retriever_name))
    for info in tqdm(data):
        corpus = {
            "doc_human": {"title": "", "text": info['doc_human']},
            "doc_llm": {"title": "", "text": info['doc_llm']},
            "doc_human_replace": {"title": "", "text": info['doc_human_replace']},
            "doc_llm_replace": {"title": "", "text": info['doc_llm_replace']}
        }
        query = {"query": info['query'], "-query": info['query']}
        results = retriever.retrieve(corpus, query)
        info['rel_human'] = results["query"]["doc_human"]
        info['rel_llm'] = results["query"]["doc_llm"]
        info['rel_human_replace'] = results["query"]["doc_human_replace"]
        info['rel_llm_replace'] = results["query"]["doc_llm_replace"]
    return data 

def save_data(data):
    print("Saving data...")
    output_path = "{}data/cocktail/{}/replace_synonyms.jsonl".format(args.root_dir, args.dataset)
    with open(output_path, 'w', encoding='utf8') as f:
        for info in tqdm(data):
            f.write(json.dumps(info) + "\n")

def load_data():
    print("Loading data...")
    output_path = "{}data/cocktail/{}/replace_synonyms.jsonl".format(args.root_dir, args.dataset)
    
    if os.path.exists(output_path) == False:
        print("File not found: {}".format(output_path))
        corpus = load_corpus()
        queries = load_queries() 
        qrels = load_qrels()
        data = conbine_data(corpus, queries, qrels)
        synonym_dict = load_synonym()

        data = replace_synonym_one(data, synonym_dict)

        data = calc_ppl_all(data)
        
        save_data(data)
        return data
              
    data = list()
    with open(output_path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            data.append(json.loads(line))
    return data




def calc_causal(data):
    delta_ppl, delta_rel = list(), list()
    for info in data:
        delta_ppl.append(info['ppl_human_replace'] - info['ppl_human'])
        delta_rel.append(info['rel_human_replace'] - info['rel_human'])
        
        delta_ppl.append(info['ppl_llm_replace'] - info['ppl_llm'])
        delta_rel.append(info['rel_llm_replace'] - info['rel_llm'])

    corr_coefficient, p_value = pearsonr(delta_ppl, delta_rel)
    print("dataset: {}".format(args.dataset))
    print(f"Pearson Correlation Coefficient: {corr_coefficient}")
    print(f"Pearson Correlation P-value: {p_value}")



if __name__ == '__main__':
   
    data = load_data()
    

    for retriever_name in ["ance"]:
        data = calc_rel(data, retriever_name)
    
        calc_causal(data)

    
    