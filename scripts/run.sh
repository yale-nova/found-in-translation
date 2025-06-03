DLRM_N_TEST_SAMPLES=1000 # 100000 used for paper
DLRM_ARGS=(--vocab-data data/dlrm/all.csv --num-test-samples "$DLRM_N_TEST_SAMPLES" --model model_weights/dlrm)

LLM_N_TEST_SAMPLES=5000 # 50000 used for paper
LLM_ARGS=(--num-test-samples "$LLM_N_TEST_SAMPLES")

HNSW_N_TEST_SAMPLES=2600
HNSW_ARGS=(--num-test-samples "$HNSW_N_TEST_SAMPLES")

run_err_eval() {
    local app_prefix=$1
    local n_test_samples=$2
    for err_rate in 1 3 5 7 10; do
        python3 attacks/language_model/eval.py --app ${app_prefix}_err$err_rate --vocab-data data/${app_prefix}/error_traces/err$err_rate.csv --num-test-samples $n_test_samples
    done
}
echo "########## HNSW Evaluation ##########"
python3 attacks/language_model/eval.py --app hnsw_nitro --vocab-data data/hnsw/all.csv "${HNSW_ARGS[@]}"
python3 attacks/language_model/eval.py --app hnsw_sgx --vocab-data data/hnsw/sgx.csv "${HNSW_ARGS[@]}"
run_err_eval "hnsw" $HNSW_N_TEST_SAMPLES

exit 0

echo "########## DLRM Evaluation ##########"
python3 attacks/language_model/eval.py --app dlrm_nitro "${DLRM_ARGS[@]}" --test-data data/dlrm/test.csv
python3 attacks/language_model/eval.py --app dlrm_sgx "${DLRM_ARGS[@]}" --test-data data/dlrm/sgx.csv
for err_rate in 1 3 5 7 10; do # Unlike LLM and HNSW, DLRM does not have unique vocab files or model weights for each error trace
    python3 attacks/language_model/eval.py --app dlrm_err$err_rate "${DLRM_ARGS[@]}" --test-data data/dlrm/error_traces/err$err_rate.csv
done
python3 attacks/language_model/eval.py --app dlrm_1_1 --vocab-data data/dlrm/all.csv --num-test-samples "$DLRM_N_TEST_SAMPLES"


echo "########## LLM Evaluation ##########"
python3 attacks/language_model/eval.py --app llm_nitro --vocab-data data/llm/all.csv "${LLM_ARGS[@]}"
python3 attacks/language_model/eval.py --app llm_sgx --vocab-data data/llm/sgx.csv "${LLM_ARGS[@]}"
run_err_eval "llm" $LLM_N_TEST_SAMPLES


