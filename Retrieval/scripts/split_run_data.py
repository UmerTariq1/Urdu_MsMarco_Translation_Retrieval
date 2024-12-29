# bash /home/butt/run_docker_cpu.sh python split_run_data.py

import pandas as pd
import os 

file_path = "../data/output/bm25/run.urdu-msmarco-passage.bm25.tsv" #from the bm25 retrieval step
n_parts = 10
output_dir = "../data/output/bm25/parts/bm25_"

# Step 1: Read the TSV file and group by the first column
df = pd.read_csv(file_path, sep='\t', header=None)
groups = df.groupby(0)

# Prepare a list to hold the grouped data
grouped_data = [group for _, group in groups]

# Step 2: Split the grouped data into 10 parts
def split_into_parts(grouped_data, n_parts=10):
    parts = [[] for _ in range(n_parts)]
    sizes = [0] * n_parts
    
    for group in grouped_data:
        # Find the part with the minimum size
        index_min = sizes.index(min(sizes))
        parts[index_min].append(group)
        sizes[index_min] += len(group)
    
    return parts

parts = split_into_parts(grouped_data, n_parts)

for i, part in enumerate(parts):
    part_df = pd.concat(part)
    # Construct the file path using the directory path from 'output_dir' and the file name
    file_path = os.path.join(output_dir, f'part_{i+1}.tsv')

    print("saving the data  to path : " + file_path)
    part_df.to_csv(file_path, sep='\t', index=False, header=False)


print("Done splitting the data into parts.")