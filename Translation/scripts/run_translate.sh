# bash run_translate.sh

python=/netscratch/butt/miniconda3/envs/thesis/bin/python

CUDA_DEVICE_NUM=0
batch_size=16
num_workers=8
num_beams=4
max_seq_len=512

model_name="ai4bharat/indictrans2-en-indic-1B"
# model_name="ai4bharat/indictrans2-en-indic-dist-200M"

continiue_from_number=-1
is_data_content_question=false

source_lang="eng_Latn"
target_lang="urd_Arab"

input_file="..data/input/english/collection/parts/part_xx.tsv"
output_dir="../data/output/urdu/parts/"


echo "Running the script"
echo "CUDA_DEVICE_NUM: $CUDA_DEVICE_NUM"
echo "Input File: $input_file"
echo "Output Directory: $output_dir"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM; $python translate.py \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --num_beams "$num_beams" \
    --max_seq_len "$max_seq_len" \
    --model_name "$model_name" \
    --output_dir "$output_dir" \
    --input_file "$input_file" \
    --source_lang "$source_lang" \
    --target_lang "$target_lang" \
    --is_data_content_question "$is_data_content_question" \
    --continiue_from_number "$continiue_from_number"