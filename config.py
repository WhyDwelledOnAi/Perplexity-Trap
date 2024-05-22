import argparse
import torch

class Config(object):
    def __init__(self) -> None:
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key , args.__dict__[key])
        self.gpu = "cuda:{}".format(self.gpu)
        # self.gpu = torch.device("cuda:{}".format(self.gpu) if torch.cuda.is_available() else "cpu")

        # [TODO]: Fill in your own data_root_dir
        self.root_dir = ""
        
        # model_path
        self.llm_dir = "{}pretrained_models/Llama-2-7b-chat-hf".format(self.root_dir) # llm
        self.compresser_dir = "{}pretrained_models/Llama-2-7B-Chat-GPTQ".format(self.root_dir) # LLMLingua

        self.bert_dir = "{}pretrained_models/bert-base-uncased".format(self.root_dir) # calc perplexity
        self.roberta_dir = "{}pretrained_models/roberta-base".format(self.root_dir)
        self.distillbert_dir = "{}pretrained_models/distilbert-base-uncased".format(self.root_dir)

        self.bert_ir_dir = "{}pretrained_models/bert-cocktail".format(self.root_dir) # bert
        self.roberta_ir_dir = "{}pretrained_models/roberta-cocktail".format(self.root_dir) # reberta
        self.sbert_dir = "{}pretrained_models/msmarco-distilbert-base-v3".format(self.root_dir) # sbert
        self.ance_dir = "{}pretrained_models/msmarco-roberta-base-ance-firstp".format(self.root_dir) # ance
        self.tasb_dir = "{}pretrained_models/msmarco-distilbert-base-tas-b".format(self.root_dir) # tas-b
        self.contriever_dir = "{}pretrained_models/contriever-msmarco".format(self.root_dir) # contriever
        self.cocondenser_dir = "{}pretrained_models/msmarco-bert-co-condensor".format(self.root_dir) # cocondenser

        self.minilm_dir = "{}pretrained_models/msmarco-MiniLM-L-12-v2".format(self.root_dir) # cross-encoder
        # dataset_path
        self.synonym_path = "{}data/cocktail/synonym_dict.json".format(self.root_dir)
        self.query_path = "{}data/cocktail/{}/queries/queries.jsonl".format(self.root_dir, self.dataset)
        self.qrels_human_path = "{}data/cocktail/{}/qrels/human.tsv".format(self.root_dir, self.dataset)
        self.qrels_llm_path = "{}data/cocktail/{}/qrels/llm.tsv".format(self.root_dir, self.dataset)
        self.corpus_human_path = "{}data/cocktail/{}/corpus/human.jsonl".format(self.root_dir, self.dataset)
        self.corpus_llm_path = "{}data/cocktail/{}/corpus/llm.jsonl".format(self.root_dir, self.dataset)
        self.corpus_merge_path = "{}data/cocktail/{}/corpus/merge.jsonl".format(self.root_dir, self.dataset)
        
        # output_path
        self.relevant_score_path = "{}data/cocktail/{}/relevant_score/init-{}.pkl".format(self.root_dir, self.dataset, self.retriever)
        self.negs_path = "{}data/cocktail/{}/qrels/hard_neg.pkl".format(self.root_dir, self.dataset)
        self.attention_score_path = "{}data/cocktail/{}/attention_score.jsonl".format(self.root_dir, self.dataset)
        self.ppl_path = "{}data/cocktail/{}/perplexity.jsonl".format(self.root_dir, self.dataset)
        
    
    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default='dl19',
                            choices=['all', 'dl19', 'scidocs', 'trec-covid'])
        
        parser.add_argument("--device", type=int, default=0, choices=[0, 1])
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--gpu", type=int,  default=0)
        parser.add_argument("--retriever", type=str, default='ance',
                            choices=['tfidf', 'bm25', "bert",'roberta', 'ance', 'tasb', 'contriever', "cocondenser"])


        parser.add_argument("--start", type=int, default=0)
        parser.add_argument("--end", type=int, default=-1)
        parser.add_argument("--map", type=str, default='none', choices=['none', 'auto', 'sequential']) 
        return parser.parse_args()
    
if __name__ == '__main__':
    config = Config()
    print(config.__dict__)