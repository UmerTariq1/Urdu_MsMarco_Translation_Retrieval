# cd /netscratch/butt/thesis2/misc
# bash /home/butt/run_docker_cpu.sh python unsplit_run_data.py

import pandas as pd
import os

input_dir="..data/output/reranker/parts"
output_file="..data/output/reranker/reranker_parts_combined.tsv"

n_parts = 10


print("Combining the files...")
print("Input directory: ", input_dir)
print("Output file: ", output_file)

def combine_files(input_dir, output_file, n_parts, base_name="run.mt5msmarco.dev.small-marco_part"):
    """ Combine part files from a directory into a single output file. """
    # mt5-base-en-msmarco_bm25_run-trec_part_1

    # Initialize an empty DataFrame to hold the combined data
    combined_df = pd.DataFrame()
    print("starting : Combining the files...")

    for i in range(1, n_parts + 1):
        # Construct the file path for each part
        part_file = os.path.join(input_dir, f"{base_name}_{i}.txt")
        # Read the part file
        part_df = pd.read_csv(part_file, sep='\t', header=None)
        # Append the part DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, part_df], ignore_index=True)

        print("files combined : ", i)

    print("Done combining the files.")

    # Write the combined DataFrame to a file
    combined_df.to_csv(output_file, sep='\t', index=False, header=False)

    print("Done writing the combined data to the output file.")


combine_files(input_dir, output_file, n_parts)
print("Done combining the files.")
