from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import copy
import math
import itertools
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import Config
args = Config()
device = args.gpu


def load_pos_pair(dataset):
    pos_pair_dir = "{}data/cocktail/{}/pos_pair.jsonl".format(args.root_dir, dataset)
    with open(pos_pair_dir, 'r', encoding='utf8') as f:
        pair_list = [json.loads(line) for line in f]
    return pair_list

class RetrievalDotLoss(nn.Module):
    def __init__(self):
        super(RetrievalDotLoss, self).__init__()
    
    def forward(self, query_emb, doc_emb):
        sim_score = - torch.mm(query_emb, doc_emb.T)
        sim_score = torch.diag(sim_score).view(-1, 1)
        return sim_score
class RetrievalCosineLoss(nn.Module):
    def __init__(self):
        super(RetrievalCosineLoss, self).__init__()
    
    def forward(self, query_emb, doc_emb):
        query_norm = torch.norm(query_emb, p=2, dim=1, keepdim=True)
        query_emb = query_emb / query_norm
        doc_norm = torch.norm(doc_emb, p=2, dim=1, keepdim=True)
        doc_emb = doc_emb / doc_norm
        sim_score = torch.diag(torch.mm(query_emb, doc_emb.T)).view(-1, 1)
        return sim_score
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# tokenizer = AutoTokenizer.from_pretrained(args.sbert_dir)
# retriever = AutoModel.from_pretrained(args.sbert_dir).to(device)
# loss_fcn = RetrievalCosineLoss()
tokenizer = None
retriever = None
loss_fcn = None

# ref_token_id = tokenizer.pad_token_id 
# sep_token_id = tokenizer.sep_token_id
# cls_token_id = tokenizer.cls_token_id
ref_token_id = None 
sep_token_id = None
cls_token_id = None


def get_embedding_query(queries):
    tokens = tokenizer(queries, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    embeddings = retriever(**tokens)[0]
    embeddings = mean_pooling(embeddings, tokens['attention_mask'])
    return embeddings
def get_embedding_document(input_ids, attention_mask):
    embeddings = retriever(input_ids=input_ids, attention_mask=attention_mask)[0]
    embeddings = mean_pooling(embeddings, attention_mask)
    return embeddings
def construct_ref_input_ids(input_ids):
    ref_input_ids = copy.deepcopy(input_ids)
    for j, input_id in enumerate(input_ids):
        for i, id in enumerate(input_id):
            if id == cls_token_id or id == sep_token_id:
                continue
            ref_input_ids[j][i] = ref_token_id
    ref_input_ids = ref_input_ids.to(device)
    return ref_input_ids
def split_batch(pair_list, batch_size):
    return_list = []
    for i in range(0, len(pair_list), batch_size):
        pair_list_batch = pair_list[i:i+batch_size]
        queries = [pair["query"] for pair in pair_list_batch]
        documents = [pair["human_doc"] for pair in pair_list_batch]
        return_list.append([queries, documents])
    return return_list
def calc_rel_score(input_ids, attention_mask, query_emb):
    doc_emb = get_embedding_document(input_ids, attention_mask)
    sim_score = loss_fcn(query_emb, doc_emb)
    return sim_score
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
def attribution(pair_list):
    lig = LayerIntegratedGradients(calc_rel_score, retriever.embeddings)
    # batch_num = math.ceil(len(pair_list) / args.batch_size)
    # print("batch_num: ", batch_num)
    attribution_list = []
    return_list = split_batch(pair_list, args.batch_size)
    for queries, documents in tqdm(return_list):
        query_emb = get_embedding_query(queries)
        
        doc_tokens = tokenizer(documents, padding=True, truncation=True, 
                               return_tensors='pt', max_length=512).to(device)
        input_ids, attention_mask = doc_tokens['input_ids'], doc_tokens['attention_mask']
        ref_input_ids = construct_ref_input_ids(input_ids)

        attributions, delta = lig.attribute(inputs=input_ids,
                        baselines=ref_input_ids,
                        additional_forward_args=(attention_mask, query_emb),
                        return_convergence_delta=True)
        attributions = summarize_attributions(attributions)
        if len(attributions.shape) == 1:
            attributions = attributions.unsqueeze(0)
        attributions = attributions.cpu().detach().numpy().tolist()
        attribution_list += attributions
    return attribution_list
def get_all_tokens(pair_list):
    pair_list_batchs = [pair_list[i:i+args.batch_size] for i in range(0, len(pair_list), args.batch_size)]
    token_list = []
    for pair_list_one_batch in pair_list_batchs:
        documents = [pair["human_doc"] for pair in pair_list_one_batch]
        doc_tokens = tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)
        for input_ids in doc_tokens['input_ids']:
            input_tokens = [tokenizer.decode(input_ids[i]) for i in range(len(input_ids))]
            token_list.append(input_tokens)
    return token_list
def remove_padding(tokens, gradients):
    new_tokens = []
    new_gradients = []
    for token, gradient in zip(tokens, gradients):
        new_token = []
        new_gradient = []
        for t, g in zip(token, gradient):
            if t != '[PAD]':
                new_token.append(t)
                new_gradient.append(g)
            else:
                assert g == 0.
        new_tokens.append(new_token)
        new_gradients.append(new_gradient)
    return new_tokens, new_gradients



def load_gradient(dataset):
    gradient_dir = "{}data/cocktail/{}/integrated_gradient_sbert.jsonl".format(args.root_dir, dataset)
    with open(gradient_dir, 'r', encoding='utf8') as f:
        gradient_list = [json.loads(line) for line in f]
    return gradient_list
def load_perplexity(dataset):
    perplexity_dir = "{}data/cocktail/{}/perplexity.jsonl".format(args.root_dir, dataset)
    id2perplexity = {}
    with open(perplexity_dir, 'r', encoding='utf8') as f:
        for line in f:
            info = json.loads(line)
            did = info['_id']
            id2perplexity[did] = info

    perplexity_list = []
    qrels_human_path = "{}data/cocktail/{}/qrels/human.tsv".format(args.root_dir, dataset)
    with open(qrels_human_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            qid, did, score = line.strip().split('\t')
            if int(score) == 0:
                continue
            perplexity_list.append(id2perplexity[did])
    return perplexity_list

def draw_scatter(gradient_list, perplexity_list):
    for i in range(len(gradient_list)):

        gradients = gradient_list[i]['gradients']
        gradients = [abs(g) for g in gradients]
        perplexity = perplexity_list[i]['ppl']
        if len(gradients) != len(perplexity):
            min_length = min(len(gradients), len(perplexity))
            gradients = gradients[:min_length]
            perplexity = perplexity[:min_length]
        gradients = np.array(gradients)
        perplexity = np.array(perplexity)
        plt.scatter(perplexity, gradients)
    plt.show()

def find_relation(dataset):
    gradient_list = load_gradient(dataset)
    for gradient in gradient_list:
        if gradient['tokens'][-1] != '[SEP]':
            gradient['tokens'] = gradient['tokens'][1:]
            gradient['gradients'] = gradient['gradients'][1:]
        else:
            gradient['tokens'] = gradient['tokens'][1:-1]
            gradient['gradients'] = gradient['gradients'][1:-1]
        
    perplexity_list = load_perplexity(dataset)
    draw_scatter(gradient_list, perplexity_list)

if __name__ == "__main__":
    datasets = ["scidocs", "trec-covid", "dl20"]
    group_names = ['8', '16', '24', '40']
    dataset_names = ["SCIDOCS", "TREC-COVID", "DL20"]
    colors = ["b", "r", "g"]

    bar_width = 0.2
    index = np.arange(len(group_names)) 

    plt.figure(figsize=(8, 6))

    for i in trange(len(datasets)):
        dataset = datasets[i]

        gradient_list = load_gradient(dataset)
        for gradient in gradient_list:
            if gradient['tokens'][-1] != '[SEP]':
                gradient['tokens'] = gradient['tokens'][1:]
                gradient['gradients'] = gradient['gradients'][1:]
            else:
                gradient['tokens'] = gradient['tokens'][1:-1]
                gradient['gradients'] = gradient['gradients'][1:-1]
        
        perplexity_list = load_perplexity(dataset)

        att_list = []
        ppl_list = []
        for j in range(len(gradient_list)):
            gradients = gradient_list[j]['gradients']
            gradients = [abs(g) for g in gradients]
            perplexity = perplexity_list[j]['ppl']
            if len(gradients) != len(perplexity):
                min_length = min(len(gradients), len(perplexity))
                gradients = gradients[:min_length]
                perplexity = perplexity[:min_length]
            att_list += gradients
            ppl_list += perplexity
        
        pd_data = pd.DataFrame({'perplexity': ppl_list, 'attribution': att_list})
        bins = [0, 12, 16, 24, 1000]
        pd_data['perplexity_group'] = pd.cut(pd_data['perplexity'], bins, labels=group_names)
        result = pd_data.groupby('perplexity_group')['attribution'].mean()
        plt.bar(index + i * bar_width, result.values, bar_width, color=colors[i], label=dataset_names[i])
        

    plt.xticks(index + 2 * bar_width, group_names)
    plt.legend()
    plt.xlabel('perplexity')
    plt.ylabel('attribution')
    plt.savefig("attribution_gradient.pdf")
    # plt.show()


