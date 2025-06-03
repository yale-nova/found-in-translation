import pandas as pd
import torch
from datasets import load_dataset
from transformers import EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

import sys
import argparse
import os
from utils import get_vocab, process_for_model, pretrained_path, pad_token_id, cls_token_id, sep_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHUFFLE_SEED = 0

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--app", type=str, help="Use case")
parser.add_argument("--train-data", type=str, help="File containing train samples (also to build vocabulary)")
parser.add_argument("--num_train_samples", type=int, help="Number of training samples")
parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of update steps between evals")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs")
parser.add_argument("--checkpointed", type=bool, default=False, help="Is there a valid checkpoint?")
parser.add_argument("--logging", type=str, default="info", help="Trainer logging level")
parser.add_argument("--logging_steps", type=int, default=10, help="Steps per log")
args = parser.parse_args(sys.argv[1:])

# Set hyperparameters
run_name = args.app
is_1_to_1 = "1_1" in run_name
batch_size = args.batch_size
learning_rate = args.learning_rate
num_train_epochs = args.num_train_epochs
train_filepath = args.train_data
resume_from_checkpoint = args.checkpointed
log_level = args.logging
logging_steps = args.logging_steps
eval_steps = args.eval_steps

num_train_samples = args.num_train_samples
num_val_samples = num_train_samples // 10
eval_batch_size = args.eval_batch_size

# directories
os.makedirs("new_model_weights", exist_ok=True)
model_dir = os.path.join("new_model_weights", run_name)

if __name__ == "__main__":
    print(f"Loading vocabulary for use case {train_filepath}...", flush=True)
    idx_to_id, id_to_idx, page_to_id, input_cols, target_cols, max_seq_len = get_vocab(train_filepath, is_1_to_1)
    encoder_max_length = max_seq_len + 2 # adding [CLS] and [SEP] tokens
    decoder_max_length = max_seq_len + 2
    col_names = input_cols + target_cols
    
    print(f"Loading train data from {train_filepath}...", flush=True)
    data = load_dataset('csv', data_files=train_filepath, sep=",")
    split = data['train'].train_test_split(shuffle=True, seed=SHUFFLE_SEED, train_size=num_train_samples)
    train_data = split['train']
    
    if num_train_samples < len(train_data):
        train_data = train_data.shuffle(seed=SHUFFLE_SEED).select(range(num_train_samples))

    print(f"Processing train data...", flush=True)
    train_data = train_data.map(
        process_for_model,
        batched=True,
        fn_kwargs={
            "input_cols": input_cols,
            "target_cols": target_cols,
            "page_to_id": page_to_id,
            "idx_to_id": idx_to_id,
            "encoder_max_length": encoder_max_length,
            "decoder_max_length": decoder_max_length,
            "is_1_to_1": is_1_to_1,
        },
        remove_columns=col_names
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"],
    )
    val_data = train_data.select(range(num_val_samples)) if num_val_samples > 0 else None
    
    print('Building model...')
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(pretrained_path, pretrained_path)

    for param in model.parameters():
        param.data = param.data.contiguous()

    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    # configure tokens
    model.config.decoder_start_token_id = cls_token_id
    model.config.eos_token_id = sep_token_id
    model.config.pad_token_id = pad_token_id

    model.encoder.resize_token_embeddings(len(page_to_id))
    model.decoder.resize_token_embeddings(len(idx_to_id))

    model.config.max_length = max_seq_len + 2
    model.config.min_length = max_seq_len
    model.config.early_stopping = True
    model.config.num_beams = 2
    print('Model config:', model.config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        logging_steps=logging_steps,
        log_level=log_level,
        log_level_replica=log_level,
        save_steps=1000,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        save_total_limit=3,
        save_safetensors=False,
        auto_find_batch_size=True,
        use_gpu=True,
        fp16=True,
    )
    print('Training arguments:', training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_model(model_dir)
    print(f"Training complete. Model saved to {model_dir}", flush=True)
