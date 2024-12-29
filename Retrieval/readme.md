The steps in this module are to be done once the translation is done. The steps are as follows:

- Convert the collection file into json (create_json_from_collection.sh)

- Create the index ( using run_docker_jvm.sh and creating_index.sh)

- Create dev small queries file (no need for this step though because we already have this file) but if not then create it.

- Create Triples file ( create_triples.sh )

- Finetune the dense retriever  ( finetune.sh  this uses gpu and takes a long time )

- Do BM25 retrieval ( bm25_retrieve.sh )

- Do BM25 evaluation ( bm25_evaluation.sh )

- Here we split the bm25 run file into X number of parts so we can run the reranking using finetuned model, faster, but ofc you can also run the reranking on the whole file at once.
> bash split_run_data.py

- Run the reranking ( deepmodel_rerank.sh ) on each part of the bm25 run file or the whole file at once.
    - the problem :  the yes/no token ids are not properly set in the **get_prediction_tokens** method in [transformers.py](in pygaggle library) , it returns the token id of a partial yes or no which is sometimes _ or etc. this conflicts with what the model outputs i.e one whole token of _yes or _no. 
    IDK why this happens and why, if model is able to generate whole tokens, then why is it (the yes no token) not in the vocabulary.
    - Temporary solution: First time just run the reranker with print statements and see the 2-3 highest tokens and print their IDs. set those IDs to yes no token id variables in **get_prediction_tokens** method**.**
    usually the no/false token id is 375 and yes/true token id is 36339
    - I am putting the transformers.py here in the scripts folder just for reference but it is to be used from the pygaggle library (/pygaggle/pygaggle/rerank/transformer.py)

- Evaluate the dense retriever ( evaluate_msmarco.sh )
