This is the code repository for the paper:
https://arxiv.org/abs/2412.12997
( Enabling Low-Resource Language Retrieval: Establishing Baselines for Urdu MS MARCO )

The model and data are available at:
https://huggingface.co/Mavkif

This repository has two folders :
Translation and Retrieval.
Each folder has its own README.md file which instructs how to recreate the data and model for the paper step by step.

There are two envioronment setups given because of different dependencies of the two folders.
each step in the sub readme file or the bash script it refers to, mentions which environment to use. if there is no mention of environment, it means the main environment is to be used. otherwise scripts that use the pyserini environment explicitly mention it.

Dataset examples can be found on the [huggingface dataset page](https://huggingface.co/datasets/urdu_msmarco).

Each of the dataset and model cards on huggingface website are explained in detail on their respective pages.