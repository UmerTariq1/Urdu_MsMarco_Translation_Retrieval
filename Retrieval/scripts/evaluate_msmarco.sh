# cd /netscratch/butt/thesis2
# bash /home/butt/run_docker_cpu.sh bash /netscratch/butt/thesis2/evaluate_msmarco.sh
# bash /home/butt/run_docker.sh -p RTXA6000 bash /netscratch/butt/thesis2/evaluate_msmarco.sh

python=/netscratch/butt/miniconda3/envs/pyserini/bin/python
echo "Running the script"


$python ms_marco_eval.py \
    ../data/urdu/qrels.dev.small.tsv \
    ..data/output/reranker/reranker_parts_combined.tsv
