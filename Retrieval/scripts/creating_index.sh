# bash creating_index.sh

python=/netscratch/butt/miniconda3/envs/pyserini/bin/python

echo "Running the script"

input_json_file="..data/output/urdu/json"
output_index_dir="..data/output/urdu/index"

language="en"

$python -m pyserini.index.lucene -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 1 \
    -input $input_json_file \
    -index $output_index_dir \
    -language $language \
    -storePositions -storeDocvectors -storeRaw


# this file has to be run with the environment which has jvm installed.
