# this file will translate using indic batch loop instead of data loader

import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time, os, re, csv, sys, ftfy, nltk, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import warnings

nltk.download('punkt')

warnings.filterwarnings("ignore", category=UserWarning)

def split_text(text, max_tokens=128):
    # convert max_tokens to int
    max_tokens = int(max_tokens)

    cleaned_text = text.strip()
    remaining_text = cleaned_text
    chunks = []
    counter = 1
    while remaining_text:
        # Split remaining text into words
        words = remaining_text.split()

        # If remaining words are within max_tokens, add the rest of the text
        if len(words) <= max_tokens:
            chunks.append(remaining_text.strip())
            break

        # Take the first max_tokens words
        chunk_words = words[:max_tokens]
        chunk = " ".join(chunk_words)

        # Search for the last sentence-ending punctuation (. ? !)
        match = re.search(r'[.!?;](?!.*[.!?])', chunk)  # Find the last valid punctuation

        if match:
            punctuation_pos = match.end()  # Position after the punctuation
            chunk = chunk[:punctuation_pos]  # Adjust the chunk to end after punctuation
        else:
            # If no punctuation, end at the last space
            last_space_pos = chunk.rfind(" ")
            if last_space_pos != -1:
                chunk = chunk[:last_space_pos]
        
        remaining_text = remaining_text[len(chunk):].lstrip()

        # Add the chunk to the list
        chunks.append(chunk.strip())

        counter += 1
    return chunks


def remove_non_english_characters(text: str) -> str:
    """
    Removes any non-English characters from the input string, 
    keeping English letters, numbers, and symbols.
    
    Args:
        text (str): The input string to clean.
        
    Returns:
        str: The cleaned string with only English letters, numbers, and symbols.
    """
    # Regular expression to match English letters, numbers, and symbols
    cleaned_text = re.sub(r'[^A-Za-z0-9\s!@#$%^&*()_+\-=\[\]{};:\'",.<>?/~`|\\]', '', text)
    return cleaned_text

class MSMarco(Dataset):
    '''
    Pytorch's dataset abstraction for MSMarco.
    '''

    def __init__(self, file_path, max_seq_len, target_language="urd"):
        self.max_seq_len = max_seq_len
        self.documents = self.load_msmarco(file_path)
        
    def __len__(self):
        return len(self.documents)

    def load_msmarco(self, file_path:str):
        '''
        Returns a list with tuples of [(doc_id, doc)].
        It uses ftfy to decode special carachters.
        Also, the special translation token ''>>target_language<<' is
        added to sentences.

        Args:
            - file_path (str): The path to the MSMarco collection file.
        '''
        documents = []
        with open(file_path, 'r', errors='ignore') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for line in tqdm(csv_reader, desc="Reading .tsv file"):

                doc_id = line[0]

                doc_lines = split_text( remove_non_english_characters( ftfy.ftfy(line[1]) ) , max_tokens=self.max_seq_len/2) # divide by 2 because its just easier for model to translate smaller sentences

                for doc in doc_lines:
                    if len(doc) > 1:
                        documents.append((doc_id, doc))
                    
        
        return documents

    def __getitem__(self,idx):
        doc_id, doc = self.documents[idx]
        return doc_id, doc
    
def initialize_model_and_tokenizer(ckpt_dir, direction,device):

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir,  trust_remote_code=True) #they have their own tokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.half()

    model.eval()

    return tokenizer, model


def prepare_dataloader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

def load_objects(model_name, input_file, batch_size, num_workers, device, max_seq_len):
    translation_loader = prepare_dataloader(MSMarco(input_file,max_seq_len), batch_size, num_workers)

    en_indic_ckpt_dir = model_name
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", device )

    ip = IndicProcessor(inference=True)


    return en_indic_model, en_indic_tokenizer, ip, translation_loader


def batch_translate(batch_sents, src_lang, tgt_lang, model, tokenizer, ip, batch_size, max_seq_len, num_beams, device, question):

    # Preprocess the batch and extract entity mappings
    batch_sents = ip.preprocess_batch(batch_sents, src_lang=src_lang, tgt_lang=tgt_lang, question=question)
    translations = []
    # Tokenize the batch and generate input encodings
    inputs = tokenizer(
        batch_sents,
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.module.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=max_seq_len, 
            num_beams=num_beams,
            num_return_sequences=1,
        )

        # generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)


    return translations


def main(model_name, input_file, output_dir, device, batch_size, num_workers, max_seq_len, num_beams, continiue_from_number,is_data_content_question, src_lang, tgt_lang):

    model, tokenizer, ip, dataset = load_objects(model_name, input_file, batch_size, num_workers, device, max_seq_len)

    output_file = output_dir + tgt_lang + '_' + input_file.split('/')[-1].split('_', 1)[1]

    print("Output file:", output_file)

    # DataParallel for multiple GPUs
    if device == "cuda":
        model = nn.DataParallel(model)
    model = model.to(device)

    start = time.time()
    counter = 0 

    start_processing = False

    write_data = []
    with open(output_file, 'a', encoding='utf-8', errors='ignore') as output:
        # get the number of lines in the file
        num_lines = sum(1 for line in open(input_file, 'r', errors='ignore'))
        print("Total number of lines in the file:", num_lines)

        for batch in tqdm(dataset, desc="Translating..."):
            doc_ids   = batch[0]
            documents = batch[1]
            counter += 1

            # below if condition is to continue from a specific doc_id in case the translation was interrupted 
            if start_processing == False and continiue_from_number!=-1 :
                for doc_id in doc_ids:
                    if int(doc_id) == continiue_from_number:
                        start_processing = True

                        doc_index = doc_ids.index(doc_id)
                        doc_ids = doc_ids[doc_index:]
                        documents = documents[doc_index:]
                        print("Starting from doc_id:", doc_id, " -- batch doc index:", doc_index)
                        break

            if start_processing == True or continiue_from_number==-1:
                src_lang, tgt_lang = src_lang, tgt_lang
                translated_documents = batch_translate(documents, src_lang, tgt_lang, model, tokenizer, ip, batch_size, max_seq_len, num_beams, device, is_data_content_question)

                for doc_id, translated_doc in zip(doc_ids, translated_documents):
                    write_data.append(doc_id + '\t' + translated_doc + '\n')

                if counter % 64 == 0:
                    output.writelines(write_data)
                    write_data = []
                    # print("Counter:", counter)
                    torch.cuda.empty_cache()


        # write the remaining data because the last section might not have 64 documents
        if len(write_data) > 0:
            output.writelines(write_data)    



    print(f"Total Time taken: {time.time() - start:.2f}s")

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Your description here.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers.')
    parser.add_argument('--num_beams', type=int, default=8, help='number of beams for beam search.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('--model_name', type=str, default="ai4bharat/indictrans2-en-indic-1B", help='model name or path')
    parser.add_argument('--output_dir', type=str, default="/local/umerbutt/thesis/data/mmarco/garbage/", help='Output directory.')
    parser.add_argument('--input_file', type=str, default="/local/umerbutt/thesis/data/mmarco/queries/dev/english_queries.dev.small.tsv", help='Input tsv file')
    parser.add_argument('--source_lang', type=str, default="eng_Latn", help='source language of the input file')
    parser.add_argument('--target_lang', type=str, default="urd_Arab", help='target language of the output file')
    parser.add_argument('--is_data_content_question', type=bool, default=False, help='Is the input file of question type strings? helps in preprocessing of input')
    parser.add_argument('--continiue_from_number', type=int, default=-1, help='doc id in input file to continue translation from')

    args = parser.parse_args()
    print("Args:", args)

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {DEVICE} for translation")
    print(f"Number of available GPUs: {num_gpus}")

    # Check the specific IDs of the GPUs
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

    main(args.model_name, args.input_file, args.output_dir, DEVICE, args.batch_size, args.num_workers, args.max_seq_len, args.num_beams, args.continiue_from_number, args.is_data_content_question, args.source_lang, args.target_lang)


    print("Done!")
