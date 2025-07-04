echo "########## Naive Bayes: DLRM Evaluation ##########"
python attacks/naive_bayes/train_eval.py --app dlrm --num-train-samples 1000000 --num-test-samples 100000
python attacks/naive_bayes/train_eval.py --app dlrm_1_1 --num-train-samples 1000000 --num-test-samples 100000

echo "########## Naive Bayes: LLM Evaluation ##########"
python attacks/naive_bayes/train_eval.py --app llm --num-train-samples 500000 --num-test-samples 50000

echo "########## Naive Bayes: HNSW Evaluation ##########"
python attacks/naive_bayes/train_eval.py --app hnsw --num-train-samples 22500 --num-test-samples 2600