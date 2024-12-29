
python=/netscratch/butt/miniconda3/envs/thesis/bin/python

echo "Running the script"

wandb_key="xxxx"
base_model="unicamp-dl/mt5-base-mmarco-v2"
tokenizer="unicamp-dl/mt5-base-mmarco-v2"

triples_path="..data/output/urdu/triples.train.small.tsv"
output_model_path=".../data/output/urdu/finetuned_model/"

max_train_data=12800000 # optional. mmarco paper also used 12.8 million data points so we used the same number
resume_from_checkpoint=""

$python ../whereever_library_is_installed_main_library_path/pygaggle/pygaggle/run/finetune_monot5.py \
    --notes "notes here to be put on wandb" \
    --base_model $base_model \
    --tokenizer $tokenizer \
    --triples_path $triples_path \
    --output_model_path $output_model_path \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --epochs 1 \
    --save_every_n_steps 5000 \
    --logging_steps 1000 \
    --max_train_data $max_train_data \
    --wandb_key $wandb_key


# --resume_from_checkpoint $resume_from_checkpoint \
# max_train_data=12800000 12.8 million data points

# this bash script calls the finetune_monot5_gpt.py python script which uses the pygaggle library. 
# so for reference i am putting finetune_monot5_gpt.py script here as well but it is to be put where the pygaggle library is installed so it can use the library functions.
