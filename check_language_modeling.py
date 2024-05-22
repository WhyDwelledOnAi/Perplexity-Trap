
from transformers import AutoModel, AutoTokenizer
from transformers import BertForMaskedLM, RobertaForMaskedLM, DistilBertForMaskedLM
import torch

from tqdm import tqdm
import json
import os

from config import Config
args = Config()

tokenizer = None
backbone = None
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")

def load_backbone(model_name):
    assert model_name in ["bert", "roberta", "ance", "tasb", "contriever", "cocondenser"]
    global tokenizer, backbone

    if model_name in ["bert", "contriever", "cocondenser"]:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
        backbone = BertForMaskedLM.from_pretrained(args.bert_dir).to(args.gpu)
        
        if model_name == "bert":
            model = AutoModel.from_pretrained(args.bert_ir_dir).to(args.gpu)
        elif model_name == "contriever":
            model = AutoModel.from_pretrained(args.contriever_dir).to(args.gpu)
        elif model_name == "cocondenser":
            model = AutoModel.from_pretrained(args.cocondenser_dir).to(args.gpu)
        elif model_name == "retromae":
            model = AutoModel.from_pretrained(args.retromae_dir).to(args.gpu)
        
        
        for name in model.state_dict():
            if name.startswith("embeddings"):
                backbone.state_dict()["bert."+name].copy_(model.state_dict()[name])
            if name.startswith("encoder"):
                backbone.state_dict()["bert."+name].copy_(model.state_dict()[name])

    if model_name in ["roberta", "ance"]:
        tokenizer = AutoTokenizer.from_pretrained(args.roberta_dir)
        backbone = RobertaForMaskedLM.from_pretrained(args.roberta_dir).to(args.gpu)
        
        if model_name == "roberta":
            model = AutoModel.from_pretrained(args.roberta_ir_dir).to(args.gpu)
        elif model_name == "ance":
            model = AutoModel.from_pretrained(args.ance_dir).to(args.gpu)

        
        for name in model.state_dict():
            if name.startswith("embeddings"):
                backbone.state_dict()["roberta."+name].copy_(model.state_dict()[name])
            if name.startswith("encoder"):
                backbone.state_dict()["roberta."+name].copy_(model.state_dict()[name])

    if model_name == "tasb":
        tokenizer = AutoTokenizer.from_pretrained(args.distillbert_dir)
        backbone = DistilBertForMaskedLM.from_pretrained(args.distillbert_dir).to(args.gpu)
        model = AutoModel.from_pretrained(args.tasb_dir).to(args.gpu)
        for name in model.state_dict():
            if name.startswith("embeddings"):
                backbone.state_dict()["distilbert."+name].copy_(model.state_dict()[name])
            if name.startswith("encoder"):
                backbone.state_dict()["distilbert."+name].copy_(model.state_dict()[name])
    print("Load {} model successfully.".format(model_name))

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
    
    if 'token_type_ids' in encoded_input.keys():
        token_type_ids = encoded_input['token_type_ids']
        token_type_ids = token_type_ids.repeat(token_type_ids.shape[1], 1)
    else:
        token_type_ids = None
    
    attention_mask = encoded_input['attention_mask']
    attention_mask = attention_mask.repeat(attention_mask.shape[1], 1)

    
    loss = []
    for i in range(0, input_ids.shape[0], args.batch_size):
        encoded_input = {'input_ids': input_ids[i:i+args.batch_size].to(args.gpu), 
                         'attention_mask': attention_mask[i:i+args.batch_size].to(args.gpu)}
        if token_type_ids is not None:
            encoded_input['token_type_ids'] = token_type_ids[i:i+args.batch_size].to(args.gpu)

        with torch.no_grad():
            model_output = backbone(**encoded_input)[0]
            pred_token_ids = [model_output[j][i+j] for j in range(model_output.shape[0])]
            pred_token_ids = torch.stack(pred_token_ids).to(args.gpu)
            loss += loss_fct(pred_token_ids, true_input_ids[i:i+args.batch_size]).cpu().tolist()
    loss = [round(l, 6) for l in loss]
    return tokens, loss

def load_dids(dataset):
    qrels_path = "{}data/cocktail/{}/qrels/human.tsv".format(args.root_dir, dataset)

    doc_ids = set()
    with open(qrels_path, 'r', encoding='utf8') as f:
        for line in f:
            data = line.strip().split('\t')
            if data[2] == 'score' or int(data[2]) == 0:
                continue
            doc_ids.add(data[1])
    return doc_ids

def test_language_modelling(model_name, dataset):
    doc_ids = load_dids(dataset)
    corpus_human_path = "{}data/cocktail/{}/corpus/human.jsonl".format(args.root_dir, dataset)
    corpus = []
    for line in open(corpus_human_path, 'r', encoding='utf8'):
        data = json.loads(line)
        if data['_id'] not in doc_ids:
            continue
        corpus.append((data['_id'], data['text']))

    
    perplexity_file = "{}data/cocktail/{}/perplexity/{}.jsonl".format(args.root_dir, dataset, model_name)
    perplexity_path = "{}data/cocktail/{}/perplexity/".format(args.root_dir, dataset)
    if not os.path.exists(perplexity_file):
        os.makedirs(perplexity_path, exist_ok=True)
        with open(perplexity_file, 'w', encoding='utf8') as f:
            pass
    for id, text in tqdm(corpus):
        tokens, loss = calc_ppl(text)
        with open(perplexity_file, 'a+', encoding='utf8') as f:
            f.write(json.dumps({"_id":id, "tokens": tokens, "ppl": loss})+'\n')
    
def calc_metric(model_name, dataset):
    perplexity_file = "{}data/cocktail/{}/perplexity/{}.jsonl".format(args.root_dir, dataset, model_name)
    loss_list = []
    with open(perplexity_file, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            loss = sum(data['ppl']) / len(data['ppl'])
            loss_list.append(loss)
    loss_mean = sum(loss_list) / len(loss_list)
    print(model_name, loss_mean)

if __name__ == '__main__':
    for model_name in ["bert", "roberta", "ance", "tasb", "contriever", "cocondenser"]:
        for dataset in ["dl19", "scidocs", "trec-covid"]:
            calc_metric(model_name, dataset) 
