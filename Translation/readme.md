- Download MS Marco dataset (https://microsoft.github.io/msmarco/Datasets.html#passage-ranking-dataset)
 there will be collection file and query files. also download the qrels train and dev files.

- split the file into multiple files so its easier to translat. no need to split query files as they are small in size. split only collection file.
> split -l x sample2.tsv parts/collection_part_


- Translate the data file by file using the script provided in the scripts folder.
> bash run_translate.sh


- After translating, it is necessary to reassemble the file, as the documents were split into sentences. Note that this division is seperate from the split done in the 2nd step. this division is so each input sentence is split into smaller part so its easier for model to translate (acc to its context window size)
use the script provided in the scripts folder
> bash combine_sentences_in_translated_file.sh

- This translation is to be done for all files i.e:
1) collection file
2) queries train file
3) queries dev file
4) queries dev-small file


- at the end make sure there are no duplicates in both query and document files. you can do this by using the first column of the file (which will be either document id or query id) as key and then removing duplicates. 