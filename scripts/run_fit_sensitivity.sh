DLRM_N_TEST_SAMPLES=$1
DLRM_ARGS=(--vocab-data data/dlrm/all.csv --num-test-samples "$DLRM_N_TEST_SAMPLES" --model model_weights/dlrm)

LLM_N_TEST_SAMPLES=$2
LLM_ARGS=(--num-test-samples "$LLM_N_TEST_SAMPLES")

HNSW_N_TEST_SAMPLES=$3
HNSW_ARGS=(--num-test-samples "$HNSW_N_TEST_SAMPLES")

# Append additional arguments, e.g., --use-cpu
shift 3
DLRM_ARGS+=("$@")
LLM_ARGS+=("$@")
HNSW_ARGS+=("$@")

run_err_eval() {
    local app_prefix=$1
    local n_test_samples=$2
    for err_rate in 1 3 5 7 10; do
        python3 attacks/fit/eval.py --app ${app_prefix}_err$err_rate --vocab-data data/${app_prefix}/error_traces/err$err_rate.csv --num-test-samples $n_test_samples
    done
}

echo "########## DLRM: Practical Considerations with $DLRM_N_TEST_SAMPLES test samples ##########"
for err_rate in 1 3 5 7 10; do # Unlike LLM and HNSW, DLRM does not have unique vocab files or model weights for each error trace
    python3 attacks/fit/eval.py --app dlrm_err$err_rate "${DLRM_ARGS[@]}" --test-data data/dlrm/error_traces/err$err_rate.csv
done

echo "########## LLM: Practical Considerations with $LLM_N_TEST_SAMPLES test samples ##########"
run_err_eval "llm" $LLM_N_TEST_SAMPLES

echo "########## HNSW: Practical Considerations with $HNSW_N_TEST_SAMPLES test samples ##########"
run_err_eval "hnsw" $HNSW_N_TEST_SAMPLES