import os
import sys
import argparse
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import train_test_split
import warnings # to suppress FutureWarnings from seaborn
 
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

ATTACK_NAME = '\\textsc{FiT}'
TITLE_LIST = ['\\textbf{DLRM}', '\\textbf{LLM}', '\\textbf{HNSW}']

# parameters from IHOP
SHUFFLE_SEED = 0
DLRM_TABLES = [1,2,5,6,8,9,13,14,17,19,20,22,23,25]

def add_cols(df, colname='targ'):
    """
    For our attack on DLRM, adds columns 'hamming_dist', 'dist_i', 'sum_dist' to result df.
    df: dataframe with columns 'colname_i' and 'pred_i' for i=1..26
    """
    df['hamming_dist'] = 0
    for i in range(1,27):
        df['hamming_dist'] += (df[f'{colname}_{i}'] != df[f'pred_{i}'])
        df[f'dist_{i}'] = (df[f'{colname}_{i}'] - df[f'pred_{i}']).abs()
    df['sum_dist'] = df[[f'dist_{i}' for i in range(1,27)]].sum(axis=1)
    return df

def add_seqlen(df):
    """
    For our attack on LLM and HNSW, adds columns 'seqlen' and 'hamming_dist' to result df.
    Trims predicted and target sequences if needed.
    df: dataframe with columns 'pred' and 'targ' containing sequences of space-separated integers.
    """
    lens = []
    hdists = []
    for i, row in df.iterrows():
        pred_lst = [int(x) for x in row['pred'].split()]
        if -1 in pred_lst:
            pred_lst = pred_lst[:pred_lst.index(-1)]
        if pred_lst[0] == -3:
            pred_lst = pred_lst[1:]
        elif -3 in pred_lst:
            pred_lst = pred_lst[:pred_lst.index(-3)]
        
        targ_lst = [int(x) for x in row['targ'].split()]
        if targ_lst[0] == -3:
            targ_lst = targ_lst[1:]
        
        if len(pred_lst) != len(targ_lst):
            if len(pred_lst) > len(targ_lst): # truncate preds
                pred_lst = pred_lst[:len(targ_lst)]
            else: # pad preds
                pred_lst += [0]*(len(targ_lst)-len(pred_lst))
        hdist = sum([0 if pred_lst[i] == targ_lst[i] else 1 for i in range(len(targ_lst))])
        hdists.append(hdist)
        lens.append(len(pred_lst))
    df['seqlen'] = lens
    df['hamming_dist'] = hdists

def get_ihdf_dlrm(fpath, idx=0, num_test_samps=100000):
    """
    For IHOP attack on DLRM, reads results into a DataFrame with target and predicted values for each table.
    fpath: path to pickle file containing output of IHOP attack.
    idx: index into list of lists of IHOP predictions.
    """
    with open(fpath,'rb') as f:
        itarg, ipred = pickle.load(f)
    num_tables = len(DLRM_TABLES)
    itarg100k = itarg[:num_test_samps*num_tables]
    ipred100k = ipred[idx][:num_test_samps*num_tables]
    
    ihdict = {}
    start_idx = 0
    for i in range(1,27):
        if i in DLRM_TABLES:
            ihdict[f'targ_{i}'] = itarg100k[start_idx::num_tables]
            ihdict[f'pred_{i}'] = ipred100k[start_idx::num_tables]
            start_idx += 1
            assert len(ihdict[f'pred_{i}']) == len(ihdict[f'targ_{i}'])
            assert len(ihdict[f'pred_{i}']) == num_test_samps, f"Table {i}: Expected {num_test_samps} preds but got {len(ihdict[f'pred_{i}'])}"
        else:
            ihdict[f'targ_{i}'] = [0] * num_test_samps
            ihdict[f'pred_{i}'] = [1] * num_test_samps
    ihdf = pd.DataFrame(data=ihdict)
    return add_cols(ihdf)

def get_ihdf_hnsw(fpath, data_path, num_train_samples, num_test_samples):
    """
    For IHOP attack on LLM and HNSW, reads results into a DataFrame with target and predicted sequences.
    """
    with open(fpath,'rb') as f:
        itarg, ipred = pickle.load(f)

    inputs = pd.read_csv(data_path)
    _, test_data = train_test_split(inputs, train_size=num_train_samples, test_size=num_test_samples, random_state=SHUFFLE_SEED, shuffle=True)
    hamming_dists = []
    lens = []
    curr_idx = 0
    
    for i, row in test_data.iterrows():
        test = [int(x) for x in row['seq'].split()]
        seq_len = len(test)
        targ = itarg[curr_idx:curr_idx+seq_len]
        pred = ipred[0][curr_idx:curr_idx+seq_len]
        assert targ == test, f'mismatch in row {i}:\n{targ}\n{test}'
        hdist = sum([0 if targ[i] == pred[i] else 1 for i in range(seq_len)])
        assert hdist <= seq_len
        hamming_dists.append(hdist)
        lens.append(seq_len)
        curr_idx += seq_len
    ihdf = pd.DataFrame(data={
        'targ': ' '.join([str(t) for t in targ]),
        'pred': ' '.join([str(p) for p in pred]),
        'hamming_dist': hamming_dists,
        'seqlen': lens,
        })
    return ihdf

def add_hamming_norm(df, is_dlrm=False):
    """
    Adds normalized hamming distance to the DataFrame.
    df: DataFrame with columns 'hamming_dist' (and 'seqlen' columns if is_dlrm is False).
    """
    if is_dlrm:
        df['hamming_norm'] = df['hamming_dist'] / 26
    else:
        df['hamming_norm'] = df['hamming_dist'] / df['seqlen']

def plot_normed_cdf(df_label_lst, xlabel="Normalized hamming distance", fname='hnsw-hamming-norm.pdf', colname='hamming_norm', legend_loc=None, xy_shift=None):
    """
    df_label_lst: list of (df,label) pairs where df has column 'hamming_norm'
    xlabel: title for x-axis
    colname: column name in df to plot
    fname: filename to save the plot
    legend_loc: location of the legend
    xy_shift: tuple to shift the legend position
    """
    fig, ax = plt.subplots(figsize=(6,3))
    for df_label in df_label_lst:
        df, label = df_label
        kwargs = {} if 'SGX' not in label else {
            'linestyle':'dotted',
            'color':'black',
            }
        ax.ecdf(df[colname], label=label, **kwargs)
        
    ax.set_ylabel('Cumulative frequency')
    ax.set_xlabel(xlabel)
    ax.legend(loc=legend_loc, bbox_to_anchor=xy_shift)
    fig.savefig(fname, bbox_inches='tight')
    # plt.show()

def plot_normed_cdf_grid(df_label_lst_by_app, titles=None, xlabel="Normalized hamming distance", fname='all-hamming-norm.pdf', colname='hamming_norm'):
    """
    df_label_lst_by_app: list of df_label_lst, where each df_label_lst is a list of (df, label) pairs for an application
    titles: list of subplot titles
    """
    n = len(df_label_lst_by_app)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 3), squeeze=False)
    for idx, df_label_lst in enumerate(df_label_lst_by_app):
        ax = axes[0, idx]
        for df, label in df_label_lst:
            kwargs = {} if 'SGX' not in label else {
                'linestyle':'dotted',
                'color':'black',
            }
            ax.ecdf(df[colname], label=label, **kwargs)
        ax.set_ylabel('Cumulative frequency')
        if idx == 0:
            ax.set_ylabel('Cumulative frequency')
        else:
            ax.set_ylabel('')
        ax.set_xlabel(xlabel)
        if titles is not None:
            ax.set_title(titles[idx])
        if idx == 1:
            ax.legend()
    fig.savefig(fname, bbox_inches='tight')
    # plt.show()

def plot_errs_grid(infile_tups, outfile_name='all-fpfn-hamming-norm.pdf'):
    """
    infile_tups: list of (infile_base, infile_suffix, name) tuples
    Each subplot will show the error CDFs for one (infile_base, infile_suffix) pair.
    """
    perc = [1, 3, 5, 7, 10]
    n = len(infile_tups)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 3))
    if n == 1:
        axes = [axes]
    for idx, (infile_base, infile_suffix, name) in enumerate(infile_tups):
        ax = axes[idx]
        is_dlrm = 'DLRM' in name
        percdfs = []
        for i in perc:
            df = pd.read_csv(f'{infile_base}{i}{infile_suffix}')
            if is_dlrm:
                add_cols(df)
            else:
                add_seqlen(df)
            add_hamming_norm(df, is_dlrm=is_dlrm)
            percdfs.append(df)
        for i, df in enumerate(percdfs):
            ax.ecdf(df['hamming_norm'], label=f'{perc[i]}% error')
        ax.set_title(name)
        ax.set_ylabel('Cumulative frequency')
        ax.set_xlabel('Normalized hamming distance')
        ax.set_xlim(0, 1.0)
        if idx == 0:
            ax.set_ylabel('Cumulative frequency')
        else:
            ax.set_ylabel('')
        if idx == 1:
            ax.legend(loc='lower center')
    fig.savefig(outfile_name, bbox_inches='tight')
    # plt.show()

def plot_durations_grid(dfs, filename, titles=None, cdf=False, nbins=50, binwidth=None):
    """
    dfs: list of DataFrames, each with columns 'Without page tracking' and 'With page tracking'
    """
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 3))
    if n == 1:
        axes = [axes]
    for i, df in enumerate(dfs):
        ax = axes[i]
        sns.histplot(
            data=(df[['Without page tracking', 'With page tracking']]/1e3),
            stat='proportion',
            bins=nbins,
            binwidth=binwidth,
            cumulative=cdf,
            element="step" if cdf else "bars",
            fill=(not cdf),
            ax=ax
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if i == 0:
            ax.set_ylabel('Cumulative frequency' if cdf else 'Frequency')
        else:
            ax.set_ylabel('')
        if i == 2:
            ax.legend(['With page tracking', 'Without page tracking'], loc='center right')
        else:
            ax.legend().remove()
        ax.set_xlabel('Inference request duration (milliseconds)')
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()

def get_results_path(app, fname):
    eval_path = os.path.join(DATA_DIR, app, 'eval', fname)
    if os.path.exists(eval_path):
        print(f"Using computed results from {eval_path}...")
        return eval_path
    fallback_path = os.path.join(DATA_DIR, app, fname)
    print(f"Computed results not found, using provided results from {fallback_path}...")
    return fallback_path


def plot_fig7(file_ext, DATA_DIR, PLOT_DIR):
    print('\nPlotting Fig. 7: Hamming distance CDFs for all attacks...')
    filename = os.path.join(PLOT_DIR, 'fig7-all-hamming-norm' + file_ext)

    dlrm_nitro = pd.read_csv(os.path.join(DATA_DIR, 'dlrm', 'eval', 'dlrm_nitro.csv'))
    add_cols(dlrm_nitro)

    dlrm_sgx = pd.read_csv(os.path.join(DATA_DIR, 'dlrm', 'eval', 'dlrm_sgx.csv'))
    add_cols(dlrm_sgx)

    hnsw_nitro = pd.read_csv(os.path.join(DATA_DIR, 'hnsw', 'eval', 'hnsw_nitro.csv'))
    add_seqlen(hnsw_nitro)

    hnsw_sgx = pd.read_csv(os.path.join(DATA_DIR, 'hnsw', 'eval', 'hnsw_sgx.csv'))
    add_seqlen(hnsw_sgx)

    llm_nitro = pd.read_csv(os.path.join(DATA_DIR, 'llm', 'eval', 'llm_nitro.csv'))
    add_seqlen(llm_nitro)

    llm_sgx = pd.read_csv(os.path.join(DATA_DIR, 'llm', 'eval', 'llm_sgx.csv'))
    add_seqlen(llm_sgx)

    ih_dlrm = get_ihdf_dlrm(get_results_path('dlrm', 'ihop_dlrm.pkl'))

    nb_dlrm = pd.read_csv(get_results_path('dlrm', 'nb_dlrm.csv')) 
    add_cols(nb_dlrm, colname='idx')

    ih_fpath = get_results_path('llm', 'ihop_llm.pkl')
    ih_llm = get_ihdf_hnsw(ih_fpath, os.path.join(DATA_DIR, 'llm', 'all.csv'), num_train_samples=500000, num_test_samples=50000)

    nb_llm = pd.read_csv(get_results_path('llm', 'nb_llm.csv'))
    add_seqlen(nb_llm)
    add_hamming_norm(nb_llm)

    ih_fpath = get_results_path('hnsw', 'ihop_hnsw.pkl')
    ih_hnsw = get_ihdf_hnsw(ih_fpath, os.path.join(DATA_DIR, 'hnsw', 'all.csv'), num_train_samples=22500, num_test_samples=2600)

    nb_hnsw = pd.read_csv(get_results_path('hnsw', 'nb_hnsw.csv'))
    add_seqlen(nb_hnsw)
    
    # Normalize hamming distances
    add_hamming_norm(ih_dlrm, is_dlrm=True)
    add_hamming_norm(dlrm_nitro, is_dlrm=True)
    add_hamming_norm(dlrm_sgx, is_dlrm=True)
    add_hamming_norm(nb_dlrm, is_dlrm=True)

    add_hamming_norm(llm_nitro)
    add_hamming_norm(llm_sgx)
    add_hamming_norm(nb_llm)
    add_hamming_norm(ih_llm)

    add_hamming_norm(hnsw_nitro)
    add_hamming_norm(hnsw_sgx)
    add_hamming_norm(ih_hnsw)
    add_hamming_norm(nb_hnsw)

    grid_inputs = [
        [(dlrm_nitro, ATTACK_NAME + ' (Nitro)'), (dlrm_sgx, ATTACK_NAME + ' (SGX)'), (nb_dlrm, 'Naive Bayes'), (ih_dlrm, 'IHOP')],
        [(llm_nitro, ATTACK_NAME + ' (Nitro)'),(llm_sgx, ATTACK_NAME + ' (SGX)'),(nb_llm, 'Naive Bayes'), (ih_llm, 'IHOP'),],
        [(hnsw_nitro, ATTACK_NAME + ' (Nitro)'),(hnsw_sgx, ATTACK_NAME + ' (SGX)'),(nb_hnsw, 'Naive Bayes'), (ih_hnsw, 'IHOP')]
    ]

    plot_normed_cdf_grid(grid_inputs, titles=TITLE_LIST, xlabel="Normalized hamming distance", fname=filename, colname='hamming_norm')
    print(f"Saved plot to {filename}.")

def plot_fig8(file_ext, DATA_DIR, PLOT_DIR):
    print('\nPlotting Fig. 8: Hamming distance CDFs for DLRM with 1-1 page-object mappings...')
    filename = os.path.join(PLOT_DIR, 'fig8-dlrm-1-1-hamming-norm' + file_ext)

    df1_1 = pd.read_csv(os.path.join(DATA_DIR, 'dlrm', 'eval', 'dlrm_1_1.csv'))
    add_cols(df1_1)

    ihdf_eval_path = get_results_path('dlrm', 'ihop_dlrm_1_1.pkl')
    ih_1_1 = get_ihdf_dlrm(ihdf_eval_path, idx=-1)
    
    nb_eval_path = get_results_path('dlrm', 'nb_dlrm_1_1.csv')
    nb_1_1 = pd.read_csv(nb_eval_path)
    add_cols(nb_1_1, colname='idx')

    add_hamming_norm(ih_1_1, is_dlrm=True)
    add_hamming_norm(nb_1_1, is_dlrm=True)
    add_hamming_norm(df1_1, is_dlrm=True)
    lst_1_1 = [(df1_1,ATTACK_NAME),(nb_1_1,'Naive Bayes') ,(ih_1_1, 'IHOP')]
    plot_normed_cdf(lst_1_1, fname=filename, legend_loc='lower center', xy_shift=(0.45, 0))
    print(f"Saved plot to {filename}.")

def plot_fig9(file_ext, DATA_DIR, PLOT_DIR):
    print('\nPlotting Fig. 9: Hamming distance CDFs with error rates in page trace...')
    filename = os.path.join(PLOT_DIR, 'fig9-all-fpfn-hamming-norm' + file_ext)
    
    err_grid_inputs = [
        (os.path.join(DATA_DIR, 'dlrm', 'eval', 'dlrm_err'), '.csv', TITLE_LIST[0]),
        (os.path.join(DATA_DIR, 'llm', 'eval', 'llm_err'), '.csv', TITLE_LIST[1]),
        (os.path.join(DATA_DIR, 'hnsw', 'eval', 'hnsw_err'), '.csv', TITLE_LIST[2]),   
    ]
    plot_errs_grid(err_grid_inputs, outfile_name=filename)
    print(f"Saved plot to {filename}.")

def plot_fig10(file_ext, DATA_DIR, PLOT_DIR):
    print('\nPlotting Fig. 10: Inference request duration CDFs...')
    filename = os.path.join(PLOT_DIR, 'fig10-all-latency-overhead' + file_ext)
    
    times_dlrm = pd.read_csv(os.path.join(DATA_DIR, 'dlrm', 'times.csv'))
    times_llm = pd.read_csv(os.path.join(DATA_DIR, 'llm', 'times.csv'))
    times_hnsw = pd.read_csv(os.path.join(DATA_DIR, 'hnsw', 'times.csv'))
    grid_inputs = [times_dlrm, times_llm, times_hnsw]
    grid_titles = TITLE_LIST
   
    plot_durations_grid(grid_inputs, filename, titles=grid_titles, cdf=True, binwidth=0.02)
    print(f"Saved plot to {filename}.")

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-ext", type=str, default=".pdf", help="File extension to plot (e.g., .pdf, .png)")
    parser.add_argument("--data-dir", type=str, default='data', help="Directory containing data files")
    parser.add_argument("--plot-dir", type=str, default='plots', help="Directory to save plots")
    parser.add_argument("--fig", type=int, help="Figure number to plot. If not specified, all figures will be plotted.")
    args = parser.parse_args(sys.argv[1:])
    
    DATA_DIR = args.data_dir
    PLOT_DIR = args.plot_dir
    
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"""
            \usepackage{mathptmx}
            \usepackage[T1]{fontenc}
            \usepackage{textcomp}
        """
    })
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 18

    if args.fig is not None:
        match args.fig:
            case 7:
                plot_fig7(args.file_ext, DATA_DIR, PLOT_DIR)
            case 8:
                plot_fig8(args.file_ext, DATA_DIR, PLOT_DIR)
            case 9:
                plot_fig9(args.file_ext, DATA_DIR, PLOT_DIR)
            case 10:
                plot_fig10(args.file_ext, DATA_DIR, PLOT_DIR)
            case _:
                print(f"Unknown figure number: {args.fig}")
    else:
        plot_fig7(args.file_ext, DATA_DIR, PLOT_DIR)
        plot_fig8(args.file_ext, DATA_DIR, PLOT_DIR)
        plot_fig9(args.file_ext, DATA_DIR, PLOT_DIR)
        plot_fig10(args.file_ext, DATA_DIR, PLOT_DIR)

    print('Done!')