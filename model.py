from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.lexical import TFIDFSearch as TFIDF
from beir.retrieval.custom_metrics import mrr, recall_cap, hole, top_k_accuracy

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import losses, models
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np
import torch
import pytrec_eval
import logging
from typing import List, Dict, Tuple
import random
from tqdm import tqdm
import pickle
import os
import copy
import json

from sentence_transformers import SentencesDataset, datasets
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.readers import InputExample
from transformers import AdamW, BertModel
from transformers import RobertaModel, RobertaConfig
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.autonotebook import trange
from typing import Dict, List, Callable, Iterable, Tuple
import logging
import time
import random
import pickle


from config import Config

args = Config()
device = args.gpu

class BERT_IR:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_ir_dir)
        self.retriever = AutoModel.from_pretrained(args.bert_ir_dir).to(device)
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def get_embd(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            model_output = self.retriever(**encoded_input)[0]
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings # [batch_size, 768]
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        queries_batch_list = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        queries_emb = [self.get_embd(queries_batch) for queries_batch in queries_batch_list]
        a = torch.cat(queries_emb, dim=0).cpu().numpy()
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        corpus = [doc['text'] for doc in corpus]
        corpus_batch_list = [corpus[i:i+batch_size] for i in range(0, len(corpus), batch_size)]
        corpus_emb = [self.get_embd(corpus_batch) for corpus_batch in corpus_batch_list]
        a = torch.cat(corpus_emb, dim=0).cpu().numpy()
        return a

class RoBERTa_IR:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(args.roberta_ir_dir)
        self.retriever = AutoModel.from_pretrained(args.roberta_ir_dir).to(device)
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def get_embd(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            model_output = self.retriever(**encoded_input)[0]
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings # [batch_size, 768]
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        queries_batch_list = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        queries_emb = [self.get_embd(queries_batch) for queries_batch in queries_batch_list]
        a = torch.cat(queries_emb, dim=0).cpu().numpy()
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        corpus = [doc['text'] for doc in corpus]
        corpus_batch_list = [corpus[i:i+batch_size] for i in range(0, len(corpus), batch_size)]
        corpus_emb = [self.get_embd(corpus_batch) for corpus_batch in corpus_batch_list]
        a = torch.cat(corpus_emb, dim=0).cpu().numpy()
        return a

class SBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(args.sbert_dir)
        self.retriever = AutoModel.from_pretrained(args.sbert_dir).to(device)
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def get_embd(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = self.retriever(**encoded_input)[0]
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings # [batch_size, 768]
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        queries_batch_list = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        queries_emb = [self.get_embd(queries_batch) for queries_batch in queries_batch_list]
        a = torch.cat(queries_emb, dim=0).cpu().numpy()
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        corpus = [doc['text'] for doc in corpus]
        corpus_batch_list = [corpus[i:i+batch_size] for i in range(0, len(corpus), batch_size)]
        corpus_emb = [self.get_embd(corpus_batch) for corpus_batch in corpus_batch_list]
        a = torch.cat(corpus_emb, dim=0).cpu().numpy()
        return a

class ANCE:
    def __init__(self):
        self.retriever = SentenceTransformer(args.ance_dir, device=args.gpu)
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        a = self.retriever.encode(queries, batch_size)
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        docs = [doc['text'] for doc in corpus]
        a = self.retriever.encode(docs, batch_size)
        return a

class TASB:
    def __init__(self):
        self.retriever = SentenceTransformer(args.tasb_dir, device=args.gpu)
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        a = self.retriever.encode(queries, batch_size)
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        docs = [doc['text'] for doc in corpus]
        a = self.retriever.encode(docs, batch_size)
        return a

class Contriever:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(args.contriever_dir)
        self.retriever = AutoModel.from_pretrained(args.contriever_dir).to(device)
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    def get_embd(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = self.retriever(**encoded_input)[0]
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings # [batch_size, 768]
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        queries_batch_list = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        queries_emb = [self.get_embd(queries_batch) for queries_batch in queries_batch_list]
        a = torch.cat(queries_emb, dim=0).cpu().numpy()
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        corpus = [doc['text'] for doc in corpus]
        corpus_batch_list = [corpus[i:i+batch_size] for i in range(0, len(corpus), batch_size)]
        corpus_emb = [self.get_embd(corpus_batch) for corpus_batch in corpus_batch_list]
        a = torch.cat(corpus_emb, dim=0).cpu().numpy()
        return a

class coCondenser:
    def __init__(self):
        self.retriever = SentenceTransformer(args.cocondenser_dir, device=args.gpu)
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        a = self.retriever.encode(queries, args.batch_size)
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        corpus = [doc['text'] for doc in corpus]
        a = self.retriever.encode(corpus, args.batch_size)
        return a

class RetroMAE:
    def __init__(self):
        self.retriever = SentenceTransformer(args.retromae_dir, device=args.gpu)
    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        a = self.retriever.encode(queries, batch_size)
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        docs = [doc['text'] for doc in corpus]
        a = self.retriever.encode(docs, batch_size)
        return a

class DRAGON:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(args.dragon_query_dir)
        self.query_encoder = AutoModel.from_pretrained(args.dragon_query_dir).to(device)
        self.doc_encoder = AutoModel.from_pretrained(args.dragon_doc_dir).to(device)
    
    def get_query_embd(self, queries):
        encoded_input = self.tokenizer(queries, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            model_output = self.query_encoder(**encoded_input).last_hidden_state[:, 0, :]
        return model_output
    
    def get_doc_embd(self, docs):
        encoded_input = self.tokenizer(docs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            model_output = self.doc_encoder(**encoded_input).last_hidden_state[:, 0, :]
        return model_output

    def encode_queries(self, queries, batch_size, **kwargs) -> np.ndarray:
        queries_batch_list = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        queries_emb = [self.get_query_embd(queries_batch) for queries_batch in queries_batch_list]
        a = torch.cat(queries_emb, dim=0).cpu().numpy()
        return a
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        corpus = [doc['text'] for doc in corpus]
        corpus_batch_list = [corpus[i:i+batch_size] for i in range(0, len(corpus), batch_size)]
        corpus_emb = [self.get_doc_embd(corpus_batch) for corpus_batch in corpus_batch_list]
        a = torch.cat(corpus_emb, dim=0).cpu().numpy()
        return a

def load_retriever(retriever_name, k_values=[1, 3, 5]):
    if retriever_name == 'tfidf':
        retriever = EvaluateRetrieval(TFIDF(), k_values=k_values)
    elif retriever_name == 'bm25':
        retriever = EvaluateRetrieval(BM25(), k_values=k_values)
    elif retriever_name == 'bert':
        retriever = EvaluateRetrieval(DRES(BERT_IR(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    elif retriever_name == 'roberta':
        retriever = EvaluateRetrieval(DRES(RoBERTa_IR(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    elif retriever_name == 'sbert':
        retriever = EvaluateRetrieval(DRES(SBERT(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='cos_sim')
    elif retriever_name == 'ance':
        retriever = EvaluateRetrieval(DRES(ANCE(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    elif retriever_name == 'tasb':
        retriever = EvaluateRetrieval(DRES(TASB(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    elif retriever_name == 'contriever':
        retriever = EvaluateRetrieval(DRES(Contriever(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    elif retriever_name == 'cocondenser':
        retriever = EvaluateRetrieval(DRES(coCondenser(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    elif retriever_name == 'retromae':
        retriever = EvaluateRetrieval(DRES(RetroMAE(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    elif retriever_name == 'dragon':
        retriever = EvaluateRetrieval(DRES(DRAGON(), batch_size=args.batch_size), 
                        k_values=k_values, score_function='dot')
    else:
        raise ValueError("retriever_name <{}> not support.".format(retriever_name))
    return retriever
            
def load_data(dataset):
    query_path = "{}data/cocktail/{}/queries/queries.jsonl".format(args.root_dir, dataset)
    qrels_human_path = "{}data/cocktail/{}/qrels/human.tsv".format(args.root_dir, dataset)
    qrels_llm_path = "{}data/cocktail/{}/qrels/llm.tsv".format(args.root_dir, dataset)
    corpus_merge_path = "{}data/cocktail/{}/corpus/merge.jsonl".format(args.root_dir, dataset)
    
    corpus, queries, qrels_human = GenericDataLoader(
            corpus_file=corpus_merge_path, 
            query_file=query_path, 
            qrels_file=qrels_human_path).load_custom()

    corpus, queries, qrels_llm = GenericDataLoader(
            corpus_file=corpus_merge_path, 
            query_file=query_path, 
            qrels_file=qrels_llm_path).load_custom()
    return corpus, queries, qrels_human, qrels_llm

def load_perplexity(dataset):
    perplexity_path = "{}data/cocktail/{}/perplexity.jsonl".format(args.root_dir, dataset)
    did2perplexity = {}
    with open(perplexity_path, 'r', encoding='utf8') as f:
        for line in f:
            ppl_data = json.loads(line)
            ppl_list = ppl_data["ppl"]
            ppl_mean = sum(ppl_list) / len(ppl_list)
            did2perplexity[ppl_data["_id"]] = ppl_mean
    return did2perplexity

def merge_dict(dict1, dict2):
    merged_dict = {}
    for key in set(dict1.keys()) | set(dict2.keys()):
        merged_dict[key] = dict()
        if key in dict1:
            merged_dict[key].update(dict1[key])
        if key in dict2:
            merged_dict[key].update(dict2[key])
    return merged_dict


debias_coef = {'bert':  {'scidocs': -0.0, 'trec-covid': -0.0, 'dl19': -0.0},
            'roberta':  {'scidocs': -0.0, 'trec-covid': -0.0, 'dl19': -0.0},
               'ance':  {'scidocs': -0.0, 'trec-covid': -0.0, 'dl19': -0.0},
               'tasb':  {'scidocs': -0.0, 'trec-covid': -0.0, 'dl19': -0.0},
         'contriever':  {'scidocs': -0.0, 'trec-covid': -0.0, 'dl19': -0.0},
        'cocondenser':  {'scidocs': -0.0, 'trec-covid': -0.0, 'dl19': -0.0}}

def calibration(retriever_name, dataset_name, results, did2perplexity):
    coef = debias_coef[retriever_name][dataset_name]
    
    results_new = copy.deepcopy(results)
    for query_id, doc_rels in results.items():
        for doc_id, rel in doc_rels.items():
            if doc_id not in did2perplexity:
                raise ValueError("doc_id <{}> not in did2perplexity.".format(doc_id))
            
            amendment = coef * did2perplexity[doc_id]
            results_new[query_id][doc_id] = rel - amendment
    return results_new


def calc_accuracy(retriever_name, dataset, k_values=[1, 3, 5]):
    corpus, queries, qrels_human, qrels_llm = load_data(dataset)
    qrels = merge_dict(qrels_human, qrels_llm)

    
    retriever = load_retriever(retriever_name, k_values=k_values)
    
    results = retriever.retrieve(corpus, queries)
    ndcg, map_, _, __ = retriever.evaluate(qrels, results, retriever.k_values)
    
    print("ndcg:", ndcg)
    
    del corpus, queries, qrels_human, qrels_llm, qrels
    del retriever
    del results


def calc_delta_metric(metric_human:dict, metric_llm:dict):
    assert metric_human.keys() == metric_llm.keys()
    delta_metric = {}
    for k in metric_human.keys():
        metric_human[k] = metric_human[k]
        metric_llm[k] = metric_llm[k]
        delta_metric[k] = 200 * (metric_human[k] - metric_llm[k]) / (metric_human[k] + metric_llm[k])
    return delta_metric

def calc_bias(retriever_name, dataset, k_values=[1, 3, 5]):
    corpus, queries, qrels_human, qrels_llm = load_data(dataset)
    
    retriever = load_retriever(retriever_name, k_values=k_values)
    
    results = retriever.retrieve(corpus, queries)
    results = calibration(retriever_name, dataset, results, load_perplexity(dataset))
    ndcg_human, map_human, _, __ = retriever.evaluate(qrels_human, results, retriever.k_values)
    ndcg_llm, map_llm, _, __ = retriever.evaluate(qrels_llm, results, retriever.k_values)
    
    delta_ndcg = calc_delta_metric(ndcg_human, ndcg_llm)
    delta_map = calc_delta_metric(map_human, map_llm)
   
    print("delta_ndcg:", delta_ndcg)
    
    del corpus, queries, qrels_human, qrels_llm
    del retriever
    del results

def check_calibration(retriever_name, dataset):
    corpus, queries, qrels_human, qrels_llm = load_data(dataset)
    qrels = merge_dict(qrels_human, qrels_llm)

    retriever = load_retriever(retriever_name, k_values=[1, 3, 5])

    rel_dir = "{}data/cocktail/{}/relevant_score/{}.jsonl".format(args.root_dir, dataset, retriever_name)
    results = dict()
    with open(rel_dir, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            info = json.loads(line)
            results.update({info['qid']: info['rel']})
            

    print("begin calibration...")
    results = calibration(retriever_name, dataset, results, load_perplexity(dataset))
    
    ndcg, map_, _, __ = retriever.evaluate(qrels, results, retriever.k_values)
    print("ndcg:", ndcg)

    ndcg_human, map_human, _, __ = retriever.evaluate(qrels_human, results, retriever.k_values)
    ndcg_llm, map_llm, _, __ = retriever.evaluate(qrels_llm, results, retriever.k_values)
    delta_ndcg = calc_delta_metric(ndcg_human, ndcg_llm)
    delta_map = calc_delta_metric(map_human, map_llm)
    print("delta_ndcg:", delta_ndcg)
    

if __name__ == '__main__':
    for dataset in ['dl19']:
        for retriever in [ 'ance']:
            print(retriever, dataset)
            # calc_accuracy(args.retriever, dataset=dataset, k_values=[3])
            # calc_bias(args.retriever, dataset, k_values=[1, 3, 5])
            check_calibration(retriever, dataset)
            print("*********************************************")


        
    