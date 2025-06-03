import pandas as pd
import torch
from datasets import load_dataset
from transformers import EncoderDecoderModel
from transformers.utils import logging

import sys
import argparse
import csv
import os.path

from utils import get_vocab, process_for_model, tensor_detokenize

logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHUFFLE_SEED = 0

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--app", type=str, help="Use case") # Possible values: {dlrm,llm,hnsw}_nitro, {dlrm,llm,hnsw}_sgx, dlrm_1_1, {dlrm,llm,hnsw}_err{1,3,5,7,10}
parser.add_argument("--model", type=str, default="", help="Model name or path")
parser.add_argument("--vocab-data", type=str, help="File to build vocabulary")
parser.add_argument("--test-data", type=str, default="", help="File containing test samples")
parser.add_argument("--num-test-samples", type=int, help="Number of test samples")
parser.add_argument("--input-batch-size", type=int, default=128, help="Batch size for input processing")
parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation")
parser.add_argument("--num-procs", type=int, default=0, help="Number of processes for generation in parallel")
args = parser.parse_args(sys.argv[1:])

run_name = args.app
is_1_to_1 = "1_1" in run_name
batch_size = args.input_batch_size
eval_batch_size = args.eval_batch_size
vocab_filepath = args.vocab_data
test_filepath = args.test_data or vocab_filepath
num_test_samples = args.num_test_samples
num_procs = args.num_procs

# directories
model_dir = args.model or f"model_weights/{run_name}" # shortcut for when app has same name as model directory (does not apply to dlrm_*) 
out_dir = f"data/{run_name.split('_')[0]}/eval"

def generate_sequence_dlrm(batch, model, num_features, id_to_idx):
    result = {}
    input_ids = batch["input_ids"].to(device)

    attention_mask = batch["attention_mask"].to(device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    label_pred_pairs = zip(batch["labels"], outputs)
    for i in range(num_features):
        result[f"pred_{i+1}"] = []
        result[f"targ_{i+1}"] = []
    
    for label, pred in label_pred_pairs:
        for i in range(num_features):
            result[f"pred_{i+1}"].append(id_to_idx[pred[i+2].item()])
            result[f"targ_{i+1}"].append(id_to_idx[label[i+1].item()])

    return result

def generate_sequence_llm_hnsw(batch, model, id_to_idx):
    inputs = batch['input_ids'].to(device)
    targets = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    preds = model.generate(
        inputs,
        attention_mask=attention_mask,
    )
    result = {
        'targ': [],
        'pred': [],
    }
    for i in range(len(inputs)):
        min_size = min(targets[i].size(0), preds[i].size(0))
        result['targ'].append(tensor_detokenize(targets[i][:min_size-1], id_to_idx))
        result['pred'].append(tensor_detokenize(preds[i][1:min_size], id_to_idx))
    return result


if __name__ == "__main__":
    # Load vocabulary, model, and test data
    print(f"Loading vocabulary for use case {vocab_filepath}...", flush=True)
    idx_to_id, id_to_idx, page_to_id, input_cols, target_cols, max_seq_len = get_vocab(vocab_filepath, is_1_to_1)
    encoder_max_length = max_seq_len + 2 # adding [CLS] and [SEP] tokens
    decoder_max_length = max_seq_len + 2
    col_names = input_cols + target_cols
    
    if test_filepath != vocab_filepath: # using separate test data file
        print(f"Loading test data from {test_filepath}...", flush=True)
        test_data = load_dataset('csv', data_files=test_filepath)['train']
    else: # using test split of vocab_filepath
        print(f"Loading test data from {vocab_filepath}...", flush=True)
        data = load_dataset('csv', data_files=vocab_filepath, sep=",")
        split = data['train'].train_test_split(shuffle=True, seed=SHUFFLE_SEED, test_size=num_test_samples)
        test_data = split['test']
    
    if num_test_samples < len(test_data):
        test_data = test_data.shuffle(seed=SHUFFLE_SEED).select(range(num_test_samples))

    print(f"Processing test data...", flush=True)
    test_data = test_data.map(
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
    test_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"],
    )

    print(f"Loading model from {model_dir}...", flush=True)
    model = EncoderDecoderModel.from_pretrained(model_dir)
    model.to(device)

    print(f"Generating sequences for {len(test_data)} test samples...", flush=True)
    if "dlrm" in run_name:
        results = test_data.map(
            generate_sequence_dlrm,
            batched=True,
            num_proc=num_procs or None,
            batch_size=eval_batch_size,
            fn_kwargs={
                "model": model,
                "num_features": max_seq_len,
                "id_to_idx": id_to_idx,
            },
        )
        result_dict = {}
        for i in range(max_seq_len):
            result_dict[f"pred_{i+1}"] = torch.flatten(results[f"pred_{i+1}"]).tolist()
            result_dict[f"targ_{i+1}"] = torch.flatten(results[f"targ_{i+1}"]).tolist()
    else:
        result_dict = test_data.map(
            generate_sequence_llm_hnsw,
            batched=True,
            num_proc=num_procs or None,
            batch_size=eval_batch_size,
            fn_kwargs={
                "model": model,
                "id_to_idx": id_to_idx,
            },
        )

    # Save predictions and ground truth to CSV
    result_df = pd.DataFrame(result_dict)

    os.makedirs(out_dir, exist_ok=True)
    out_filename = run_name + '.csv' # test_filepath[test_filepath.index("data/") + len("data/"):].replace("/", "_")
    out_path = os.path.join(out_dir, out_filename)
    result_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}", flush=True)
