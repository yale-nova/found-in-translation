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

echo "########## DLRM: Attack Efficacy with $DLRM_N_TEST_SAMPLES test samples ##########"
python3 attacks/fit/eval.py --app dlrm_nitro "${DLRM_ARGS[@]}" --test-data data/dlrm/test.csv
python3 attacks/fit/eval.py --app dlrm_sgx "${DLRM_ARGS[@]}" --test-data data/dlrm/sgx.csv

echo "########## LLM: Attack Efficacy with $LLM_N_TEST_SAMPLES test samples ##########"
python3 attacks/fit/eval.py --app llm_nitro --vocab-data data/llm/all.csv "${LLM_ARGS[@]}"
python3 attacks/fit/eval.py --app llm_sgx --vocab-data data/llm/sgx.csv "${LLM_ARGS[@]}"

echo "########## HNSW: Attack Efficacy with $HNSW_N_TEST_SAMPLES test samples ##########"
python3 attacks/fit/eval.py --app hnsw_nitro --vocab-data data/hnsw/all.csv "${HNSW_ARGS[@]}"
python3 attacks/fit/eval.py --app hnsw_sgx --vocab-data data/hnsw/sgx.csv "${HNSW_ARGS[@]}"

echo "########## DLRM 1-1 page-to-object mapping baseline: Attack Efficacy with $DLRM_N_TEST_SAMPLES test samples ##########"
python3 attacks/fit/eval.py --app dlrm_1_1 --vocab-data data/dlrm/all.csv --num-test-samples "$DLRM_N_TEST_SAMPLES"
