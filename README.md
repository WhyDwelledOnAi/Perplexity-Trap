# Perplexity-Trap: PLM-Based Retrievers Overrate Low Perplexity Documents

The implementation of the paper: "Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents"

Previous studies have found that PLM-based retrieval models exhibit a preference for LLM-generated content, assigning higher relevance scores to these documents even when their semantic quality is comparable to human-written ones. This phenomenon, known as source bias, threatens the sustainable development of the information access ecosystem. However, the underlying causes of source bias remain unexplored. In this paper, we explain the process of information retrieval with a causal graph where the PLM-based retrievers learn perplexity features and take it as a basis for relevance estimation, causing source bias if ranking the documents with low perplexity. Theoretical analysis further reveals that the phenomenon stems from the positive correlation between the gradients of the loss functions in language modeling task and retrieval task. Based on the analysis, a causal-inspired inference-time debiasing method is proposed, called **C**ausal **D**iagnosis and **C**orrection (CDC), which first diagnoses the bias effect of the perplexity and then separate the bias effect from the overall estimated relevance score. Experimental results across three domains demonstrate the superior debiasing effectiveness of CDC, emphasizing the validity of our proposed explanatory framework.

## File Description
The experiments in this paper are based on the codes below: 
> calc_attention.py: attention-based input attribution method.\\
> calc_perplexity.py: calculate token-level and document-level perplexity by BERT.\\
> cacl_rel.py: calculate estimated relevance scores of retrieval models at top@k.
> calc_rel_pos: select all relevant query-document pairs.
> check_language_modeling.py: estimate the language modeling ability for each retriever.
> config.py: experiment settings or hyperparamters.
> find_relation_temp_ppl.py:  evaluate the correlationship between document perplexity and estimated relevance scores with varying temperatures.
> human_eval.py: sample (query-human\_document-LLM\_document) triples to receive human evaluation and aggregate evaluation results.
> integrated_gradients.py: integrated gradients input attribution method.
> model.py: build retrievers, evaluate their accuracy and source bias, and de-bias via CDC.
> two_stage_reg.py: evaluate the magnitude and significance of causal effect of document perplexity on estimated relevance scores.
