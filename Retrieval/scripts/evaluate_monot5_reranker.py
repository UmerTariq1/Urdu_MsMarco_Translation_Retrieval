import argparse
import pandas as pd
from tqdm import tqdm
from pygaggle.rerank.transformer import MonoT5
from pygaggle.rerank.base import Query, Text
import jsonlines
from transformers import (
)

def load_corpus_as_dict(path):
    with open(path, 'r', encoding='utf-8') as file:
        corpus = {}
        for line in file:
            key, value = line.strip().split('\t', 1)  # Split into key and value at the first tab
            # convert the keys to int
            key = int(key)
            corpus[key] = value
    return corpus

def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    if '.json' in path:
        with jsonlines.open(path) as reader:
            for obj in tqdm(reader):
                id = int(obj['id'])
                corpus[id] = obj['contents']
    else: #Assume it's a .tsv
        # corpus = pd.read_csv(path, sep='\t', header=None, index_col=0)[1].to_dict()
        corpus = load_corpus_as_dict(path)
    return corpus


def load_run(path):
    print('Loading run...')
    run = pd.read_csv(path, delim_whitespace=True, header=None)
    run = run.groupby(0)[1].apply(list).to_dict()
    return run


def load_queries(path):
    print('Loading queries...')
    queries = pd.read_csv(path, sep='\t', header=None, index_col=0)
    queries = queries[1].to_dict()
    return queries
# 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='unicamp-dl/mt5-base-multi-msmarco', type=str, required=False,
                        help="Reranker model.")
    parser.add_argument("--initial_run", default=None, type=str, required=True,
                        help="Initial run to be reranked.")
    parser.add_argument("--corpus", default=None, type=str, required=True,
                        help="Document collection.")
    parser.add_argument("--output_run", default=None, type=str, required=True,
                        help="Path to save the reranked run.")
    parser.add_argument("--queries", default=None, type=str, required=True,
                        help="Path to the queries file.")

    args = parser.parse_args()
    print("Args : ", args)

    # IMP : Add model path to MonoT5 class as prediction tokens
    model = MonoT5(args.model_name_or_path)

    print("Model loaded :", args.model_name_or_path)

    run = load_run(args.initial_run)
    print("Runs loaded :", args.initial_run, " : Total queries in the run :", len(run))

    corpus = load_corpus(args.corpus)
    print("Corpus loaded :", args.model_name_or_path ," : Total lines in the corpus:", len(corpus))

    queries = load_queries(args.queries)
    print("Queries loaded :", args.queries, " : Total queries :", len(queries))

    part_number = args.initial_run.split("/")[-1].split("_")[-1].split(".")[0]
    part_number = "part_" + part_number
    print("Part number is : " + part_number)

    # Run reranker
    trec = open(args.output_run + '-trec_' + part_number + '.txt','w')
    marco = open(args.output_run + '-marco_' + part_number + '.txt','w')

    print("Trec filename : ", args.output_run + '-trec_' + part_number + '.txt')
    print("Marco filename : ", args.output_run + '-marco_' + part_number + '.txt')

    for idx, query_id in enumerate(tqdm(run.keys())):
        query = Query(queries[query_id])
        texts = [Text(corpus[doc_id], {'docid': doc_id}, 0) for doc_id in run[query_id]]
        reranked = model.rerank(query, texts)

        for rank, document in enumerate(reranked):
            trec.write(f'{query_id}\tQ0\t{document.metadata["docid"]}\t{rank+1}\t{document.score}\t{args.model_name_or_path}\n')
            marco.write(f'{query_id}\t{document.metadata["docid"]}\t{rank+1}\n')

    trec.close()
    marco.close()
    print("Done writing in the part number !" + part_number)

    print("The arguments were : ", args)

if __name__ == "__main__":
    main()