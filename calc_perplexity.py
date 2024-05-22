import json
from tqdm import tqdm, trange
import torch
from transformers import BertForMaskedLM, AutoTokenizer
import copy
import os
# import warnings
# warnings.filterwarnings('ignore')


from config import Config

args = Config()

tokenizer = None
model = None
loss_fct = None
# tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
# model = BertForMaskedLM.from_pretrained(args.bert_dir).to(args.gpu)
# loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")


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

    # 按照batch_size分组
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

def write_perplexity(corpus_merge_path, ppl_path):
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    global model 
    model= BertForMaskedLM.from_pretrained(args.bert_dir).to(args.gpu)
    global loss_fct 
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")

    corpus = []
    with open(corpus_merge_path, 'r', encoding='utf8') as f:
        for line in f:
            corpus.append(json.loads(line))
    
    with open(ppl_path, 'a+', encoding='utf8') as f:
        for i in trange(len(corpus)):
            if i < args.start:
                continue
            if args.end > 0 and i > args.end:
                break
            info = corpus[i]
            did = info['_id']
            # print("Calc perplexity of document {}.".format(did))
            text = info['text']
            tokens, loss = calc_ppl(text)
            f.write(json.dumps({'_id':did, 'tokens': tokens, 'ppl': loss}) + '\n')

def check_perplexity(corpus_merge_path, ppl_path):
    dids = set()
    with open(ppl_path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            info = json.loads(line)
            dids.add(info['_id'])
    missing_dids = []
    with open(corpus_merge_path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            info = json.loads(line)
            if info['_id'] not in dids:
                missing_dids.append(info['_id'])
    return missing_dids

if __name__ == '__main__':
    # corpus_merge_path = "{}data/cocktail/{}/corpus/replace_all.jsonl".format(args.root_dir, args.dataset)
    # ppl_path = "{}data/cocktail/{}/perplexity_replace_all.jsonl".format(args.root_dir, args.dataset)
    # write_perplexity(corpus_merge_path, ppl_path)
    if args.dataset == 'all':
        for i, dataset in enumerate(['dl19', 'scidocs', 'trec-covid']):
            print("dataset {} is processing. [{}/16]".format(dataset, i+1))
            corpus_merge_path = "{}data/cocktail/{}/merge.jsonl".format(args.root_dir, dataset)
            ppl_path = "{}data/cocktail/{}/perplexity.jsonl".format(args.root_dir, dataset)
            write_perplexity(corpus_merge_path, ppl_path)
            
    else:
        corpus_merge_path = args.corpus_merge_path
        ppl_path = args.ppl_path
        write_perplexity(corpus_merge_path, ppl_path)

    corpus_merge_path = args.corpus_merge_path
    ppl_path = args.ppl_path
    missing_dids = check_perplexity(corpus_merge_path, ppl_path)
    print(missing_dids)
        
        
