python=/netscratch/butt/miniconda3/envs/thesis/bin/python

qrels_dev_small_file="..data/urdu/qrels.dev.small.tsv"
bm25_retrieval_file="..data/output/urdu/bm25/run.msmarco-passage.bm25.dev.small.tsv"

echo "Running the script"

$python pygaggle/tools/scripts/msmarco/msmarco_passage_eval.py \
    $qrels_dev_small_file \
    $bm25_retrieval_file

# Results will look like
#####################
# MRR @10: xxxx
# QueriesRanked: yyyy
#####################


# this doesnt give the recall. so you can also use evaluate_msmarco.sh