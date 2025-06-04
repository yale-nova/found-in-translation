import pandas as pd
import numpy as np
import torch

pretrained_path = "prajjwal1/bert-tiny"

pad_token_id = 0
unk_token_id = 100
cls_token_id = 101
sep_token_id = 102
mask_token_id = 103
special_token_ids = [pad_token_id, unk_token_id, cls_token_id, sep_token_id, mask_token_id]

max_seq_len = 512

def get_vocab(vocab_filename, is_1_to_1=False):
    """
    Returns model vocabularies for the use case [app].
    """
    if "dlrm" in vocab_filename:
        return get_dlrm_vocab(vocab_filename, is_1_to_1)
    # else "llm" or "hnsw"
    return get_llm_hnsw_vocab(vocab_filename)
        

def get_llm_hnsw_vocab(vocab_filename):
    # Set up page/index/token id mappings
    input_col = 'encseq'
    target_col = 'seq'

    inputs = pd.read_csv(vocab_filename)

    # building vocabs
    unique_indices = set()
    unique_pages = set()
    max_in_len = 0
    min_in_len = max_seq_len
    max_out_len = 0
    for i, row in inputs.iterrows():
        node_ids = list(map(int, row[target_col].split()))
        hashed_ids = list(map(int, row[input_col].split()))
        unique_indices.update(node_ids)
        unique_pages.update(hashed_ids)
        max_in_len = max(max_in_len, len(hashed_ids))
        min_in_len = min(min_in_len, len(hashed_ids))
        max_out_len = max(max_out_len, len(node_ids))

    # index vocab
    max_idx = len(unique_indices)
    id_to_idx = { v:-(i+1) for i,v in enumerate(special_token_ids) }
    idx_to_id = { -(i+1):v for i,v in enumerate(special_token_ids) }
    add_to_idx = 0
    for i,v in enumerate(unique_indices):
        if i in special_token_ids:
            idx_to_id[v] = max_idx + add_to_idx
            id_to_idx[max_idx + add_to_idx] = v
            add_to_idx += 1        
        else:
            idx_to_id[v] = i
            id_to_idx[i] = v
    idx_vocab_size = len(id_to_idx)
    assert idx_vocab_size == len(idx_to_id), f"{idx_vocab_size} != {len(idx_to_id)}"

    # page vocab
    max_idx = len(unique_pages)
    page_to_id = {}
    for i,v in enumerate(special_token_ids):
        assert -(i+1) not in unique_pages
        page_to_id[-(i+1)] = v
    add_to_idx = 0
    for i,v in enumerate(unique_pages):
        if i in special_token_ids:
            page_to_id[v] = max_idx + add_to_idx
            add_to_idx += 1
        else:
            page_to_id[v] = i
    
    seq_len = max(max_in_len, max_out_len)
    return idx_to_id, id_to_idx, page_to_id, [input_col], [target_col], seq_len

def get_dlrm_vocab(vocab_filename, is_1_to_1=False):
    """
    Returns model vocabularies for the DLRM use case, i.e., mappings between sequence values and token IDs.
    """
    # Set up page/index/token ID mappings
    num_features = 26
    input_cols = [f'page_{i+1}' for i in range(num_features)]
    target_cols = [f'idx_{i+1}' for i in range(num_features)]

    inputs = pd.read_csv(vocab_filename)

    # index vocab (ground-truth object-level accesses)
    unique_indices = pd.unique(inputs[target_cols].values.ravel('K'))
    max_idx = len(unique_indices)
    id_to_idx = { v:-(i+1) for i,v in enumerate(special_token_ids) }
    idx_to_id = { -(i+1):v for i,v in enumerate(special_token_ids) }

    add_to_idx = 0
    for i,v in enumerate(unique_indices):
        if i in special_token_ids:
            idx_to_id[v] = max_idx + add_to_idx
            id_to_idx[max_idx + add_to_idx] = v
            add_to_idx += 1
        else:
            idx_to_id[v] = i
            id_to_idx[i] = v
    idx_vocab_size = len(id_to_idx)
    assert idx_vocab_size == len(idx_to_id), f"{idx_vocab_size} != {len(idx_to_id)}"

    # page vocab (observable page-level accesses)
    if is_1_to_1:
        unique_pages = pd.DataFrame()
        for input_col, target_col in zip(input_cols, target_cols):
            pairs = pd.DataFrame({
                'page': inputs[input_col],
                'idx': inputs[target_col],
                'tbl': int(input_col.split('_')[1]),
            }).drop_duplicates()
            unique_pages = pd.concat([unique_pages, pairs])
        unique_pages = list(zip(unique_pages['page'], unique_pages['idx'], unique_pages['tbl']))
    else:
        unique_pages = pd.unique(inputs[input_cols].values.ravel('K'))
    if -2 in unique_pages:
        unique_pages = np.delete(unique_pages, 0)
    assert -2 not in unique_pages
    max_page = len(unique_pages)
    page_to_id = { -(i+1):v for i,v in enumerate(special_token_ids)}
    add_to_idx = 0
    for i,v in enumerate(unique_pages):
        if i in special_token_ids:
            page_to_id[v] = max_page + add_to_idx
            add_to_idx += 1
        else:
            page_to_id[v] = i
    if not is_1_to_1:
        page_to_id = dict(sorted(page_to_id.items())) # sort keys in page_to_id

    return idx_to_id, id_to_idx, page_to_id, input_cols, target_cols, num_features

def get_closest_pages(pages, sorted_pages): # pages_in_dict is a tensor
    """
    Returns indices of closest pages in [sorted_pages], in tensor with same shape as [pages]
    """
    return torch.searchsorted(sorted_pages, pages)

def process_for_model(batch, input_cols, target_cols, page_to_id, idx_to_id, encoder_max_length, decoder_max_length, is_1_to_1=False):
    """
    Converts a batch of sequences into [input_ids], [attention_mask], and [labels] for model inference.
    """
    num_features = len(input_cols)
    if is_1_to_1: # dlrm1-1
        return process_for_model_dlrm_1_1(batch, page_to_id, idx_to_id, num_features, encoder_max_length, decoder_max_length)
    elif num_features > 1: # dlrm
        return process_for_model_dlrm(batch, page_to_id, idx_to_id, num_features, encoder_max_length)
    else: # llm/hnsw
        return process_for_model_llm_hnsw(batch, input_cols[0], target_cols[0], page_to_id, idx_to_id, encoder_max_length)

def process_for_model_dlrm_1_1(batch, page_to_id, idx_to_id, num_features, encoder_max_length, decoder_max_length):
    bsize = len(batch["page_1"])

    batch["input_ids"] = [
        [cls_token_id] + [page_to_id[(batch[f"page_{i+1}"][j],batch[f"idx_{i+1}"][j],i+1)] for i in range(num_features)] + [sep_token_id]
        for j in range(bsize)]
    batch["attention_mask"] = [[1] * encoder_max_length] * bsize
    batch["labels"] = [
        [cls_token_id] + [idx_to_id[batch[f"idx_{i+1}"][j]] for i in range(num_features)] + [sep_token_id]
        for j in range(bsize)]
    for j in range(bsize):
        assert len(batch["input_ids"][j]) == encoder_max_length
        assert len(batch["labels"][j]) == decoder_max_length
    return batch

def process_for_model_dlrm(batch, page_to_id, idx_to_id, num_features, encoder_max_length):
    bsize = len(batch["page_1"])
    
    input_ids = [[cls_token_id] + [sep_token_id]*(num_features + 1) for _ in range(bsize)]
    for i in range(num_features):
        sorted_pages = list(page_to_id.keys())
        pages_i = torch.tensor(batch[f"page_{i+1}"])
        closest_pages_indices = get_closest_pages(pages_i, torch.tensor(sorted_pages)).tolist()
        for j in range(bsize):
            page_ij = sorted_pages[min(closest_pages_indices[j], len(sorted_pages)-1)]
            input_ids[j][i+1] = page_to_id[page_ij]
    batch["input_ids"] = input_ids
    batch["attention_mask"] = [[1] * encoder_max_length] * bsize
    batch["labels"] = [
        [cls_token_id] + [idx_to_id[batch[f"idx_{i+1}"][j]] for i in range(num_features)] + [sep_token_id]
        for j in range(bsize)]
    return batch

def batch_tokenize(batch_vals, tok_to_id, bsize, encoder_max_length):
    seq_batch = []
    attn_batch = []
    for j in range(bsize):
        seq = [cls_token_id] + [0] * (encoder_max_length - 1)
        attn = [1] + [0] * (encoder_max_length - 1)
        
        toks = batch_vals[j].split()
        seq_len = len(toks)
        for i in range(seq_len):
            seq[i+1] = tok_to_id[int(toks[i])]
            attn[i+1] = 1
        seq[seq_len+1] = sep_token_id
        attn[seq_len+1] = 1
        
        assert seq.count(sep_token_id) == 1, f"seq[{j}] = {seq}\ntoks = {toks}\nseq_len = {seq_len}"
        
        seq_batch.append(seq)
        attn_batch.append(attn)
    return seq_batch, attn_batch

def process_for_model_llm_hnsw(batch, input_col, target_col, page_to_id, idx_to_id, encoder_max_length):
    bsize = len(batch[input_col])
    inputs, attn = batch_tokenize(batch[input_col], page_to_id, bsize, encoder_max_length)
    batch["input_ids"] = inputs
    batch["attention_mask"] = attn
    labels, _ = batch_tokenize(batch[target_col], idx_to_id, bsize, encoder_max_length)
    batch["labels"] = labels
    return batch

def tensor_detokenize(tens, id_to_tok):
    tok_lst = []
    for x in tens.tolist():
        if x == sep_token_id:
            break
        tok_lst.append(str(id_to_tok[x]))
    return ' '.join(tok_lst)
