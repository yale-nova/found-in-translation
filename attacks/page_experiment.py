import time
import sys
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from config import PAGE_ACCESS_MODE_TO_DATASET_FOLDER, DLRM_TABLES, SHUFFLE_SEED
from exp_params import ExpParams
from experiment import run_attack

def _compute_Faux(trace, n):
    Faux = np.zeros((n, n))
    m = np.histogram2d(trace[1:], trace[:-1], bins=(range(n+1), range(n+1)))[0] / (len(trace) - 1)
    for j in range(n):
        if np.sum(m[:, j]) > 0:
            Faux[:, j] = m[:, j] / np.sum(m[:, j])
    return Faux

def _get_data_adv(data_path, num_train_samples, num_test_samples, input_col, target_col):
    """
    Load the auxiliary dataset from [data_path] and build the F_aux matrix from values in [target_col].
    """
    all_filename = os.path.join(data_path, "all.csv")
    
    print("Loading datasets...")
    inputs = pd.read_csv(all_filename)
    
    print(f"Building auxiliary dataset...")
    unique_pages = set()              
    unique_indices = set()
    for i, row in inputs.iterrows():
        obj_ids = list(map(int, row[target_col].split()))
        unique_indices.update(obj_ids)
        page_ids = list(map(int, row[input_col].split()))
        unique_pages.update(page_ids)
    page_to_idx = {page: idx for idx, page in enumerate(unique_pages)}
    n = len(unique_indices)
    chosen_kw_indices = list(range(n))
    
    data_adv = [[x] for x in unique_indices]

    print(f"nkw = {n}. Building F_aux from training data...")
    train_data, test_data = train_test_split(inputs, train_size=num_train_samples, test_size=num_test_samples, random_state=SHUFFLE_SEED, shuffle=True)
    
    trace = []
    for i, row in train_data.iterrows():
        inference_request = [int(v) for v in row[target_col].split()]
        trace += inference_request
    Faux = _compute_Faux(trace, n)
    del inputs, train_data
    
    return data_adv, Faux, chosen_kw_indices, page_to_idx, test_data

def get_traces_real(test_data, page_to_idx, input_col, target_col):
    traces = []
    real_queries = []
    for _, row in test_data.iterrows():
        traces += [(v, [page_to_idx[int(v)]]) for v in row[input_col].split()]
        real_queries += [int(v) for v in row[target_col].split()]
    return traces, real_queries

def generate_accesses(gen_params, num_train_samples, num_test_samples):
    """
    For HNSW, LLM: generate adversary's auxiliary dataset and real object-level accesses.
    """
    data_path = PAGE_ACCESS_MODE_TO_DATASET_FOLDER[gen_params['dataset']]
    input_col = 'encseq'    
    target_col = 'seq'

    data_adv, Faux, chosen_kw_indices, page_to_idx, test_data = _get_data_adv(data_path, num_train_samples, num_test_samples, input_col, target_col)
    full_data_adv = {'dataset': data_adv,
                     'keywords': chosen_kw_indices,
                     'frequencies': Faux,
                     'mode_query': gen_params['mode_query']}
    print('Generated auxiliary dataset.')

    assert len(test_data) == num_test_samples, f"Expected {num_test_samples} test samples, got {len(test_data)}"
    print("Generating page trace and real object accesses...")
    traces, real_queries = get_traces_real(test_data, page_to_idx, input_col, target_col)
    observations = {
        'trace_type': 'ap_unique',
        'traces': traces,
        'ndocs': len(data_adv)
    }
    print('Done.')

    return full_data_adv, observations, real_queries

def _get_id_closest_page(page, page_to_id):
    if page in page_to_id:
        return page_to_id[page]
    sorted_pages = list(page_to_id.keys()) # assume sorted keys
    
    if page > sorted_pages[-1]:
        return page_to_id[sorted_pages[-1]]
    # binary search for closest page
    lo = 0
    hi = len(sorted_pages) - 1
    while lo <= hi:
        cur = (hi + lo) // 2
        if page < sorted_pages[cur]:
            hi = cur - 1
        elif page > sorted_pages[cur]:
            lo = cur + 1
        else:
            return page_to_id[sorted_pages[cur]]
    # lo == hi + 1
    if sorted_pages[lo] - page < page - sorted_pages[hi]:
        return page_to_id[sorted_pages[lo]]
    return page_to_id[sorted_pages[hi]]    

def get_traces_real_dlrm(tables, test_data, page_to_idx, value_to_idx, is_1_to_1=False):
    traces = []
    real_queries = []
    for _, row in test_data.iterrows():
        for i in tables:
            token_id = f"{row[f'page_{i}']}_{row[f'idx_{i}']}{f'_{i}' if is_1_to_1 else ''}"
            traces.append((token_id, [_get_id_closest_page(row[f'page_{i}'], page_to_idx)])) # (token_id, [doc_ids])
            real_queries.append(value_to_idx[(row[f'idx_{i}'],i)])
    return traces, real_queries
    
def _get_data_adv_dlrm(data_path, num_train_samples, num_test_samples, tables):
    all_filename = os.path.join(data_path, "all.csv")
    input_cols = [f'page_{i}' for i in tables]
    
    print("Loading datasets...")
    inputs = pd.read_csv(all_filename)
    train_data, _ = train_test_split(inputs, train_size=num_train_samples, test_size=num_test_samples, random_state=SHUFFLE_SEED, shuffle=True)
    unique_pages = pd.unique(inputs[input_cols].values.ravel('K'))
    page_to_idx = {value: idx for idx, value in enumerate(unique_pages)}
    page_to_idx = dict(sorted(page_to_idx.items()))
    
    print(f"Building auxiliary dataset...")              
    page_to_indices = {}
    for i, row in train_data.iterrows():
        for j in tables:
            page = row[f'page_{j}']
            if page not in page_to_indices:
                page_to_indices[page] = set()
            page_to_indices[page].add(row[f'idx_{j}'])
    data_adv = list(map(lambda x: list(x), page_to_indices.values())) # kws (entries) in each doc (page)

    unique_indices = set()
    for i in tables:
        entries = inputs[f'idx_{i}']
        unique_indices.update((value, i) for value in entries)
    print(f"len(train_data) = {len(train_data)}.")
    n = len(unique_indices)
    value_to_idx = {value: idx for idx, value in enumerate(unique_indices)}
    chosen_kw_indices = list(range(n))

    print(f"nkw = {n}. Building F_aux from training data...")
    Faux = np.zeros((n, n))
    m = np.zeros((n, n))
    trace = []
    for i, row in train_data.iterrows():
        inference_request = [value_to_idx[(row[f'idx_{i}'],i)] for i in tables]
        trace += inference_request
    m = np.histogram2d(trace[1:], trace[:-1], bins=(range(n+1), range(n+1)))[0] / (len(trace) - 1)
    for j in range(n):
        if np.sum(m[:, j]) > 0:
            Faux[:, j] = m[:, j] / np.sum(m[:, j])

    del inputs, train_data
    return data_adv, Faux, chosen_kw_indices, value_to_idx, page_to_idx

def generate_dlrm_accesses(gen_params, num_train_samples, num_test_samples):
    data_path = PAGE_ACCESS_MODE_TO_DATASET_FOLDER[gen_params['dataset']]
    test_filename = os.path.join(data_path, 'test.csv')

    data_adv, Faux, chosen_kw_indices, value_to_idx, page_to_idx = _get_data_adv_dlrm(data_path, num_train_samples, num_test_samples, DLRM_TABLES)
    full_data_adv = {'dataset': data_adv,
                     'keywords': chosen_kw_indices,
                     'frequencies': Faux,
                     'mode_query': gen_params['mode_query']}
    print('Generated auxiliary dataset.')

    test_data = pd.read_csv(test_filename)
    assert len(test_data) == num_test_samples, f"Expected {num_test_samples} test samples, got {len(test_data)}"
    print(f"nqr = {len(test_data) * len(DLRM_TABLES)}. Generating page trace and real object accesses...")
    traces, real_queries = get_traces_real_dlrm(DLRM_TABLES, test_data, page_to_idx, value_to_idx, is_1_to_1='1_1' in gen_params['dataset'])
    assert len(real_queries) == len(test_data) * len(DLRM_TABLES)
    observations = {
        'trace_type': 'ap_unique',
        'traces': traces,
        'ndocs': len(data_adv)
    }

    return full_data_adv, observations, real_queries

def run_page_access_experiment(exp_param, seed, num_train_samples, num_test_samples, debug_mode=False):
    v_print = print if debug_mode else lambda *a, **k: None

    t0 = time.time()
    np.random.seed(seed)
    if exp_param.gen_params['dataset'] not in PAGE_ACCESS_MODE_TO_DATASET_FOLDER: # original ihop experiment
        raise ValueError("Dataset {:s} not supported for page access experiments".format(exp_param.gen_params['dataset']))
    
    if 'dlrm' in exp_param.gen_params['dataset']:  # page access experiment: DLRM
        full_data_adv, observations, real_and_dummy_queries = generate_dlrm_accesses(exp_param.gen_params, num_train_samples, num_test_samples)
        print("Generated observations ({:.1f} secs)".format(time.time() - t0))
    else:  # page access experiment: LLM, HNSW
        full_data_adv, observations, real_and_dummy_queries = generate_accesses(exp_param.gen_params, num_train_samples, num_test_samples)
        print("Generated observations ({:.1f} secs)".format(time.time() - t0))
    
    keyword_predictions_for_each_query = run_attack(exp_param.att_params['name'], obs=observations, aux=full_data_adv, exp_params=exp_param)
    v_print("\nDone running attack ({:.1f} secs)".format(time.time() - t0))
    time_exp = time.time() - t0
    
    return keyword_predictions_for_each_query, real_and_dummy_queries, time_exp
   
    
def print_exp_to_run(parameter_dict, n_runs):
    for key in parameter_dict:
        print('  {:s}: {}'.format(key, parameter_dict[key]))
    print("* Number of runs: {:d}".format(n_runs))


if __name__ == "__main__":

    os.system('mesg n')

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, help="Use case") # Possible values: dlrm, dlrm_1_1, llm, hnsw
    parser.add_argument("--num-train-samples", type=int, help="Number of train samples for auxiliary data")
    parser.add_argument("--num-test-samples", type=int, help="Number of test samples for observed data")
    args = parser.parse_args(sys.argv[1:])

    exp_params = ExpParams()
    exp_params.set_general_params(dataset=args.app, nkw=100, nqr=10_000, freq='file',
                                  mode_ds='same', mode_fs='same', mode_kw='rand', mode_query='markov')
    exp_params.set_defense_params('none')
    attack_list = [('ihop', {'mode': 'Freq', 'niters': 1000, 'pfree': 0.25})]
    niter_list = [0, 10, 100, 1000]

    np.set_printoptions(precision=4)
    print(exp_params)

    seed = 1
    print("Seed: ", seed)

    for i_att, (att, att_p) in enumerate(attack_list):
        exp_params.set_attack_params(att, **att_p)
        exp_params.att_params['niter_list'] = niter_list
        preds, targs, time_exp = run_page_access_experiment(exp_params, seed, args.num_train_samples, args.num_test_samples, debug_mode=True)
        out_dir = os.path.join(PAGE_ACCESS_MODE_TO_DATASET_FOLDER[exp_params.gen_params['dataset']], 'eval')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"ihop_{exp_params.gen_params['dataset']}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump((targs, preds), f)
        print("Saved results to {:s}".format(output_path))

