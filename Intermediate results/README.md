# Intermediate results

Due to the space limitation of anonymous Github, we only upload the key examples of SCIDOCS dataset for easliy reproducing our experiments during the review process. Upon accepted, we will upload all the constructed datasets to Google Drive or other shared space for downloading.

For all PLM-based retrievers, we select results from ANCE as examples.

For all sampling temperature, we select results from temp=0.0 as examples. 

Also, we only provide 10 pieces of examples of document perplexity because the full file is too large.

The content in each file is as follows:
> {retriever_name}.json: the top@k retrieved documents with their estimated relevance scores.

> pos_rel_{retriever_name}.json: the estimated relevance scores of (query, human_document, LLM_document) triplets.

> tmp{temperature}-ori.json: LLM-generated documents with different sampling temperatures via rewriting.

> perplexity.json: documents' tokens with their perplexity calculated with BERT or similar models.

