echo "########## IHOP: DLRM Evaluation ##########"
python attacks/USENIX22-ihop-code/page_experiment.py --app dlrm --num-train-samples 1000000 --num-test-samples 100000
python attacks/USENIX22-ihop-code/page_experiment.py --app dlrm_1_1 --num-train-samples 1000000 --num-test-samples 100000

echo "########## IHOP: LLM Evaluation ##########"
python attacks/USENIX22-ihop-code/page_experiment.py --app llm --num-train-samples 500000 --num-test-samples 50000

echo "########## IHOP: HNSW Evaluation ##########"
python attacks/USENIX22-ihop-code/page_experiment.py --app hnsw --num-train-samples 22500 --num-test-samples 2600
