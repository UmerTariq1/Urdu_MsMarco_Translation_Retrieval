
python="/netscratch/butt/miniconda3/envs/pyserini/bin/python"

dev_small_queries="..data/input/urdu/queries.dev.small.tsv"
index_folder_path="..data/output/urdu/index"
language="en"
output_path="..data/output/urdu/bm25/run.msmarco-passage.bm25.dev.small.tsv"

echo "Running the script"

$python -m pyserini.search.lucene \
    --topics $dev_small_queries \
    --index $index_folder_path \
    --language $language \
    --output $output_path \
    --bm25 \
    --output-format msmarco \
    --hits 1000  \
    --k1 0.6 \
    --b 0.8


# the output file format is:
# Query ID \t Document ID \t Rank

# you need environment with pyserini installed to run this script

# configure the k1 and b values by testing out on different values
