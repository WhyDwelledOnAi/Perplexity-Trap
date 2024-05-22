import json
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, DistilBertModel
import matplotlib.pyplot as plt


from config import Config

args = Config()
device = args.gpu


def load_pos_pair(dataset):
    pos_pair_dir = "{}data/cocktail/{}/pos_pair.jsonl".format(args.root_dir, dataset)
    with open(pos_pair_dir, 'r', encoding='utf8') as f:
        pair_list = [json.loads(line) for line in f]
    return pair_list

class RetrievalLoss(nn.Module):
    def __init__(self):
        super(RetrievalLoss, self).__init__()
    
    def forward(self, query_emb, doc_emb):
        sim_score = - torch.mm(query_emb, doc_emb.T)
        sim_score = torch.diag(sim_score).view(-1, 1)
        return sim_score
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained(args.contriever_dir)
retriever = AutoModel.from_pretrained(args.contriever_dir).to(device)
loss_fcn = RetrievalLoss()

def get_gradient(queries, documents):
    query_tokens = tokenizer(queries, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    query_emb = retriever(**query_tokens)[0]
    query_emb = mean_pooling(query_emb, query_tokens['attention_mask'])
    
    doc_tokens = tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    input_embeds = retriever.embeddings.word_embeddings(doc_tokens['input_ids'])
    input_embeds = Variable(input_embeds, requires_grad=True)
    doc_emb = retriever(inputs_embeds=input_embeds, attention_mask=doc_tokens['attention_mask'])[0]
    doc_emb = mean_pooling(doc_emb, doc_tokens['attention_mask'])

    loss = loss_fcn(query_emb, doc_emb)
    input_gradients_norm_lists = []
    for i in range(len(loss)):
        loss[i].backward(retain_graph=True)
        input_gradients = input_embeds.grad[i]
        input_gradients_norm = torch.linalg.norm(input_gradients, ord=2, dim=1)
        input_gradients_norm_lists.append(input_gradients_norm.tolist())
        
    return input_gradients_norm_lists
    
def get_all_gradient(pair_list):
    pair_list_batchs = [pair_list[i:i+args.batch_size] for i in range(0, len(pair_list), args.batch_size)]

    gradient_list = []
    for pair_list_one_batch in tqdm(pair_list_batchs):
        queries = [pair["query"] for pair in pair_list_one_batch]
        documents = [pair["human_doc"] for pair in pair_list_one_batch]
        gradient_list += get_gradient(queries, documents)
    return gradient_list

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

def calc_gridient(dataset, pair_list):
    output_dir = "{}data/cocktail/{}/input_gradient_contriever.jsonl".format(args.root_dir, dataset)
    all_tokens = get_all_tokens(pair_list)
    all_gradients = get_all_gradient(pair_list)
    all_tokens, all_gradients = remove_padding(all_tokens, all_gradients)
    with open(output_dir, 'w', encoding='utf8') as f:
        for i, (tokens, gradients) in enumerate(zip(all_tokens, all_gradients)):
            f.write(json.dumps({"_id": i, "tokens": tokens, "gradients": gradients}) + '\n')

def load_gradient(dataset):
    gradient_dir = "{}data/cocktail/{}/input_gradient_contriever.jsonl".format(args.root_dir, dataset)
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
    with open(args.qrels_human_path, 'r', encoding='utf8') as f:
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
        tokens_gradients = gradient_list[i]['tokens']
        tokens_perplexity = perplexity_list[i]['tokens']

        gradients = gradient_list[i]['gradients']
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
        
        
if __name__ == '__main__':
    pair_list = load_pos_pair(args.dataset)
    calc_gridient(args.dataset, pair_list)
    # find_relation(args.dataset)

    