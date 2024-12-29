# bash combine_sentences_in_translated_file.sh

python=/netscratch/butt/miniconda3/envs/thesis/bin/python

input_file="../data/output/urdu/parts/xx.tsv"
output_file="../data/output/urdu/parts_combioned/xx.tsv"

echo "Running the script"

$python combine_translated_sentences.py \
    --output_file "$output_file" \
    --input_file "$input_file"

