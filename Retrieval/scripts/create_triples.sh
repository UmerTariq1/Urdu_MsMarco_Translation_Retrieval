
python=/netscratch/butt/miniconda3/envs/thesis/bin/python

echo "Running the script"

query_file="..data/input/urdu/queries.train.tsv"
collection_file="..data/input/urdu/collection.tsv"
triples_qrels_id_file="..data/input/urdu/triples.train.ids.small.tsv"

output_file="..data/output/urdu/triples.train.small.tsv"

$python misc/create_triples_file.py \
    --query_file $query_file \
    --collection_file $collection_file \
    --triples_file $triples_qrels_id_file \
    --output_file $output_file