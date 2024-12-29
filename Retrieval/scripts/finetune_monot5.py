
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    MT5ForConditionalGeneration,
    EarlyStoppingCallback
)
import wandb
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonoT5Dataset(Dataset):
    def __init__(self, queries: List[str], passages: List[str], labels: List[str]):
        self.queries = queries
        self.passages = passages
        self.labels = labels

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        text = f'Query: {self.queries[idx]} Document: {self.passages[idx]} Relevant:'
        return text, self.labels[idx]

def load_data(triples_path: str, valid_size: float = 0.1, max_train_data=-1, make_data_small=0) -> Tuple[MonoT5Dataset, MonoT5Dataset]:
    """Load and split data into train and validation sets."""
    queries, passages, labels = [], [], []
    
    logger.info(f"Loading data from {triples_path}")
    logger.info(f"max_train_data {max_train_data}")


    start_from_data = -1 
    end_on_data = -1


    logger.info(f"--> Ignoring max_train_data for now")
    logger.info(f"--> Start from data: {start_from_data}")
    logger.info(f"--> End on data: {end_on_data}")

    current_data_count = 0
    with open(triples_path, 'r', encoding="utf-8") as f:
        for line_number, line in enumerate(f):

            # Normal training code , use this for normal training. the other one is because i couldnt run it for longer time
            if max_train_data > 0 and current_data_count >= max_train_data:
                logger.info(f"Max data limit reached while reading the input file. Breaking the loop.")
                logger.info(f"Current data count: {current_data_count}")
                break

            
            # for 2nd or 3rd day of training, on newer data
            if start_from_data!=-1 and current_data_count < start_from_data:
                current_data_count = current_data_count + 2  # 2 samples per line
                continue
            if end_on_data!=-1 and current_data_count > end_on_data:
                break

            # ---- general code applicable for both normal and 2nd day of training, on newer data
            query, positive, negative = line.strip().split("\t")
            queries.extend([query, query])
            passages.extend([positive, negative])
            labels.extend(['yes', 'no'])

            current_data_count = current_data_count + 2  # 2 samples per line

    
    logger.info(f"Number of samples read from the file: {len(queries)}")

    # make_data_small is the number of samples to keep for debugging
    if make_data_small > 0:
        queries = queries[:make_data_small]
        passages = passages[:make_data_small]
        labels = labels[:make_data_small] 

    # Split into train and validation
    train_queries, val_queries, train_passages, val_passages, train_labels, val_labels = train_test_split(
        queries, passages, labels, test_size=valid_size, random_state=42
    )
    
    train_dataset = MonoT5Dataset(train_queries, train_passages, train_labels)
    val_dataset = MonoT5Dataset(val_queries, val_passages, val_labels)
    
    print("--> \n An example of train_dataset : ", train_dataset[0])

    print("/n --------------- ---------------")
    logger.info(f"Loaded {len(train_dataset)} training samples .... and ... loaded {len(val_dataset)} validation samples")

    return train_dataset, val_dataset

def smart_batching_collate_text_only(batch, tokenizer, device):
    texts = []
    labels = []
    
    for text, label in batch:
        texts.append(text)
        labels.append(label)

    tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
    tokenized['labels'] = tokenizer(labels, padding=True, return_tensors='pt')['input_ids']

    for name in tokenized:
        tokenized[name] = tokenized[name].to(device)

    return tokenized

def create_model_and_tokenizer(args):
    """Create model and tokenizer with configured settings."""
    logger.info(f"Loading model from {args.base_model}")
    
    config = AutoConfig.from_pretrained(
        args.base_model,
        dropout=0.1,
        attention_dropout=0.1,
        encoder_layerdrop=0.1,
        decoder_layerdrop=0.1
    )
    
    model = MT5ForConditionalGeneration.from_pretrained(args.base_model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    
    return model, tokenizer

def get_training_args(args, num_training_steps):
    """Create training arguments with proper configuration."""
    return Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,

        do_train=True,
        do_eval=True,

        save_strategy="steps",
        eval_strategy="steps",  # Align with save strategy

        save_steps=args.save_every_n_steps,
        eval_steps=args.save_every_n_steps,
        logging_steps=args.logging_steps,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        # lr_scheduler_type='linear'
        lr_scheduler_type='cosine',
        weight_decay=1e-4,

        num_train_epochs=args.epochs,

        optim="adafactor",
        warmup_ratio=0.1,
        seed=42,

        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        greater_is_better=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=str, help="Notes for wandb logging")
    parser.add_argument("--base_model", default='unicamp-dl/mt5-base-mmarco-v2', type=str)
    parser.add_argument("--tokenizer", default='unicamp-dl/mt5-base-mmarco-v2', type=str)
    parser.add_argument("--triples_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--save_every_n_steps", default=10000, type=int)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--per_device_train_batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--max_train_data", default=-1, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--wandb_key", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    logger.info(f"max_train_data : {args.max_train_data}")

    wandb.login(key=args.wandb_key)
    wandb.init(
        project="msmarco-finetunings",
        config={
            "model_name": args.base_model,
            "triples_path": args.triples_path,
            "output_model_path": args.output_model_path,
            "notes": args.notes,
            "All args": vars(args)
        }
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"--> Device: {device}")
    
    # Create output directory
    Path(args.output_model_path).mkdir(parents=True, exist_ok=True)
    logger.info("--> Output directory created")
    
    # Load data
    train_dataset, val_dataset = load_data(args.triples_path, max_train_data=int(args.max_train_data))
    logger.info("--> Data loaded")
    logger.info(f"--> Train dataset size : {len(train_dataset)}")
    logger.info(f"--> Val dataset size : {len(val_dataset)}")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args)
    model = model.to(device)
    logger.info("--> Model and tokenizer created")
    
    # Calculate total training steps for warmup
    num_training_steps = len(train_dataset) * args.epochs // (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    logger.info(f"--> Total training steps: {num_training_steps}")

    # Get training arguments
    training_args = get_training_args(args, num_training_steps)
    logger.info("--> Training arguments created")

    # Create data collator
    data_collator = lambda batch: smart_batching_collate_text_only(batch, tokenizer, device)
    logger.info("--> Data collator created")
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    logger.info("--> Trainer initialized")
    
    # Train
    logger.info("Starting training...")

    if args.resume_from_checkpoint:
        logger.info(f"--> Resuming from checkpoint: {args.resume_from_checkpoint}..")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        logger.info(f"Training from scratch..")
        trainer.train()
    
    # Save final model
    trainer.save_model(args.output_model_path)
    wandb.finish()

if __name__ == "__main__":
    main()