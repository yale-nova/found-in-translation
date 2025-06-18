import pandas as pd
import numpy as np
import csv
import sys
import os
import time
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import math
import random

PAGE_ACCESS_MODE_TO_DATASET_FOLDER = {
    'dlrm': 'data/dlrm/',
    'dlrm_1_1': 'data/dlrm/',
    'llm': 'data/llm/',
    'hnsw': 'data/hnsw/',
}
SHUFFLE_SEED = 0
DLRM_NUM_TABLES = 26


class NaiveBayesHistogram:
    def __init__(self, find_closest=False):
        self.pages_to_indices = {}
        self.input_col = 'encseq'
        self.target_col = 'seq'
        self.find_closest = find_closest
        self.min_page = 0
    
    def fit(self, xy_train):
        for i, row in xy_train.iterrows():
            encseq = row[self.input_col].split()
            seq = row[self.target_col].split()
            assert len(encseq) == len(seq), f"Length mismatch: {len(encseq)} != {len(seq)}"
            for i, page in enumerate(encseq):
                page_int = int(page)
                if page_int not in self.pages_to_indices:
                    self.pages_to_indices[page_int] = []
                self.pages_to_indices[page_int].append(int(seq[i]))
        self.min_page = min(self.pages_to_indices.keys())
    
    def predict(self, xy_test, output_file):
        with open(output_file, "w") as out_file:
            out_writer = csv.writer(out_file)
            out_writer.writerow(['targ', 'pred', 'hamming_dist', 'seqlen'])
            for _, row in xy_test.iterrows():
                encseq = row[self.input_col].split()
                target = list(map(int, row[self.target_col].split()))
                prediction = []
                for page in encseq:
                    page = int(page)
                    if page in self.pages_to_indices:
                        prediction.append(np.random.choice(self.pages_to_indices[page]))
                    else: # find nearest page
                        if self.find_closest:
                            nearest_page = min(self.pages_to_indices.keys(), key=lambda x: abs(x - int(page)))
                        else:
                            nearest_page = self.min_page
                        prediction.append(np.random.choice(self.pages_to_indices[nearest_page]))
                out_writer.writerow([row[self.target_col], ' '.join(list(map(str, prediction))), sum([1 for i in range(len(target)) if target[i] != prediction[i]]), len(target)])

class NaiveBayesHistogramDLRM:
    def __init__(self, num_features, lookup_header, sparse_header, is_1_to_1=False, find_closest=False):
        self.num_features = num_features
        self.incols = lookup_header
        self.outcols = sparse_header
        self.pages_to_indices_per_lookup = [set() for _ in range(num_features)] if is_1_to_1 else [{} for _ in range(num_features)]
        self.is_1_to_1 = is_1_to_1
        self.find_closest = find_closest
    
    def fit(self, xy_train):
        if self.is_1_to_1:
            for i, row in xy_train.iterrows():
                for j in range(self.num_features):
                    page_accessed = row[self.incols[j]]
                    target_index = row[self.outcols[j]]
                    key = (page_accessed, target_index) if self.is_1_to_1 else page_accessed
                    self.pages_to_indices_per_lookup[j].add(key)
        else:
            for i in range(self.num_features):
                pages_accessed = pd.unique(xy_train[self.incols[i]])
                for page_accessed in pages_accessed:
                    target_indices = xy_train[xy_train[self.incols[i]] == page_accessed][self.outcols[i]]
                    self.pages_to_indices_per_lookup[i][page_accessed] = target_indices
    
    def predict(self, xy_test, output_file):
        data = []
        pred_colname = 'pred'
        if self.is_1_to_1:
            for _, row in xy_test.iterrows():
                result_row = {}
                for i in range(self.num_features):
                    page_accessed = row[self.incols[i]]
                    target_index = row[self.outcols[i]]
                    key = (page_accessed, target_index)
                    if not key in self.pages_to_indices_per_lookup[i]:
                        pred_j = 0
                    else:
                        pred_j = target_index
                    result_row[f'{pred_colname}_{i+1}'] = pred_j
                    result_row[self.outcols[i]] = target_index
                data.append(result_row)
        else:
            for _, row in xy_test.iterrows():
                result_row = {}
                for i in range(self.num_features):
                    page_accessed = row[self.incols[i]]
                    if not page_accessed in self.pages_to_indices_per_lookup[i]: # page not in training data => use closest page
                        page_accessed = min(self.pages_to_indices_per_lookup[i].keys(), key=lambda x: abs(x - page_accessed)) \
                            if self.find_closest else 0
                    target_indices = self.pages_to_indices_per_lookup[i][page_accessed]
                    result_row[f'{pred_colname}_{i+1}'] = target_indices.sample(n=1, replace=True).values[0]
                    result_row[self.outcols[i]] = row[self.outcols[i]]
                data.append(result_row)
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

    def find_closest_key(self, heap, page_accessed):
        keys = sorted(heap)
        pos = self.binary_search(keys, page_accessed)
        if pos == 0:
            return keys[0]
        if pos == len(keys):
            return keys[-1]
        before = keys[pos - 1]
        after = keys[pos]
        if after[0] - page_accessed < page_accessed - before[0]:
            return after
        else:
            return before
    
    def binary_search(self, keys, page_accessed):
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid][0] < page_accessed:
                lo = mid + 1
            else:
                hi = mid
        return lo

class NaiveBayes:
    def __init__(self):
        self.y_val_freq = Counter() # tracks Pr(Y)
        self.x_freq_per_y = defaultdict(Counter) # tracks Pr(X | Y)
        self.vocab = set()
        self.total_samples = 0

    def fit(self, X, Y):
        """
        X: list of lists of input page accesses
        Y: list of lists of ground-truth output object accesses
        """
        for x_seq, y_seq in zip(X, Y):
            for x, y in zip(x_seq, y_seq):
                self.y_val_freq[y] += 1
                self.x_freq_per_y[y][x] += 1
                self.vocab.add(x)
                self.total_samples += 1

        # Precompute class priors and likelihoods with Laplace smoothing
        self.class_priors = {
            y: count / self.total_samples for y, count in self.y_val_freq.items()
        }

        self.likelihoods = {}
        for y in self.y_val_freq:
            total_y = sum(self.x_freq_per_y[y].values())
            self.likelihoods[y] = {
                x: (self.x_freq_per_y[y][x] + 1) / (total_y + len(self.vocab))
                for x in self.vocab
            }

    def predict_single(self, x_seq, sample):
        predictions = []
        for x in x_seq:
            probs = {}
            for y in self.y_val_freq:
                log_prior = math.log(self.class_priors[y])
                likelihood = self.likelihoods[y].get(
                    x, 1 / (sum(self.x_freq_per_y[y].values()) + len(self.vocab))
                )
                log_likelihood = math.log(likelihood)
                probs[y] = log_prior + log_likelihood

            if sample: # Sample from normalized probabilities
                max_log = max(probs.values())
                exp_probs = {y: math.exp(logp - max_log) for y, logp in probs.items()}
                total = sum(exp_probs.values())
                norm_probs = {y: p / total for y, p in exp_probs.items()}

                labels = list(norm_probs.keys())
                weights = list(norm_probs.values())
                predictions.append(random.choices(labels, weights=weights, k=1)[0])
            else: # Argmax
                best_y = max(probs, key=probs.get)
                predictions.append(best_y)
        return predictions
    
    def predict(self, X, sample=True):
        return [self.predict_single(x_seq, sample=sample) for x_seq in X]    
    
    def save_predictions(self, output_path, predictions, targets):
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['targ', 'pred'])
            for target, pred in zip(targets, predictions):
                writer.writerow([' '.join(map(str, target)), ' '.join(map(str, pred))])

def process_sequences(df, input_col='encseq', output_col='seq'):
    X_seq = df[input_col].apply(lambda s: [int(tok) for tok in s.strip().split()]).tolist()
    Y_seq = df[output_col].apply(lambda s: [int(tok) for tok in s.strip().split()]).tolist()

    return X_seq, Y_seq


def get_output_path(app):
    return os.path.join(PAGE_ACCESS_MODE_TO_DATASET_FOLDER[app], 'eval', f'nb_{app}.csv')

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, help="Use case") # Possible values: dlrm, dlrm_1_1, llm, hnsw
    parser.add_argument("--num-train-samples", type=int, help="Number of train samples for auxiliary data")
    parser.add_argument("--num-test-samples", type=int, help="Number of test samples for observed data")
    args = parser.parse_args(sys.argv[1:])
    
    print("Evaluating Naive Bayes with arguments:", sys.argv[1:])

    if args.app not in PAGE_ACCESS_MODE_TO_DATASET_FOLDER:
        raise ValueError(f"Unknown app: {args.app}. Available options: {list(PAGE_ACCESS_MODE_TO_DATASET_FOLDER.keys())}")
    
    np.random.seed(SHUFFLE_SEED)
    random.seed(SHUFFLE_SEED)
    t = time.time()

    xy = pd.read_csv(os.path.join(PAGE_ACCESS_MODE_TO_DATASET_FOLDER[args.app], 'all.csv'))
    xy_train, xy_test = train_test_split(xy, train_size=args.num_train_samples, test_size=args.num_test_samples, random_state=SHUFFLE_SEED)
    output_path = get_output_path(args.app)

    if 'dlrm' in args.app:
        in_colnames = [f'page_{i+1}' for i in range(DLRM_NUM_TABLES)]
        out_colnames = [f'idx_{i+1}' for i in range(DLRM_NUM_TABLES)]
        is_1_to_1 = '1_1' in args.app
        model = NaiveBayesHistogramDLRM(DLRM_NUM_TABLES, in_colnames, out_colnames, is_1_to_1=is_1_to_1, find_closest=not is_1_to_1)
        model.fit(xy_train)
        print(f"Fitted model: {time.time() - t:.2f} seconds", flush=True)
        
        model.predict(xy_test, output_path)
        print(f"Predicted: {time.time() - t:.2f} seconds", flush=True)

    elif args.app == 'hnsw': 
        model = NaiveBayesHistogram(find_closest=False)
        model.fit(xy_train)
        print(f"Fitted model: {time.time() - t:.2f} seconds", flush=True)

        model.predict(xy_test, output_path)
        print(f"Predicted: {time.time() - t:.2f} seconds", flush=True)        

    else: # llm
        x_train, y_train = process_sequences(xy_train)
        model = NaiveBayes() # faster than NaiveBayesHistogram but with higher memory usage and lower accuracy
        model.fit(x_train, y_train)
        print(f"Fitted model: {time.time() - t:.2f} seconds", flush=True)
        
        x_test, y_test = process_sequences(xy_test)
        y_pred = model.predict(x_test)
        print(f"Predicted: {time.time() - t:.2f} seconds", flush=True)

        model.save_predictions(output_path, y_pred, y_test)

    print("Done. Wrote predictions to", output_path, flush=True)
 