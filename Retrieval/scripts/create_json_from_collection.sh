# bash create_json_from_collection.sh

python=/netscratch/butt/miniconda3/envs/thesis/bin/python


input_file="..data/input/urdu/collection.tsv"
output_dir="..data/output/urdu/json"

echo "Running the script"

$python pygaggle/tools/scripts/msmarco/convert_collection_to_jsonl.py \
    --collection-path "$input_file" \
    --output-folder "$output_dir"
