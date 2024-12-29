# need environment with pyserini 

python=/netscratch/butt/miniconda3/envs/pyserini/bin/python

model_name="unicamp-dl/mt5-base-mmarco-v2" # the correct model name that was finetuned

queries="..data/input/urdu/queries.dev.small.tsv"

run_file="../data/output/bm25/parts/bm25_part_xx.tsv" # change the xx to the part number

corpus="..data/input/urdu/collection.tsv"

output_run="..data/output/reranker/parts/monoT5_reranker_" # just pass the base name, the script will add the part number

echo "Running the script"
echo "model name : " $model_name
echo "run file : " $run_file
echo "corpus : " $corpus
echo "queries : " $queries
echo "output run : " $output_run


$python /pygaggle/evaluate_monot5_reranker.py \
    --model_name_or_path $model_name \
    --initial_run $run_file \
    --corpus $corpus \
    --queries $queries \
    --output_run $output_run


# this bash script calls the python script to rerank the initial run file using the monoT5 model.
# the file called is evaluate_monot5_reranker.py which uses the pygaggle library to rerank the initial run file. hence that 
# python file has to be placed where it can access the pygaggle library. i am putting it in the scripts folder for reference.