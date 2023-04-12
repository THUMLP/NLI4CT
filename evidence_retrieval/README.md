## Environment
torch==1.7.1
tqdm==4.64.1
scikit-learn==1.0.2
transformers==4.24.0

## Train & Evaluate
Please download the necessary pre-trained models from huggingface and Then save it to "biolinkbert/".
```
# For training models, you may directly run the command:
# biolinkbert_new_sentpooling_sentinter
python run_fold0.py --model biolinkbert_new_sentpooling_sentinter

# biolinkbert_new_sentpooling_sentinter_block
python run_fold0.py --model biolinkbert_new_sentpooling_sentinter_block

# biolinkbert_sent_tokenpooling_sentinter
python run_fold0.py --model biolinkbert_sent_tokenpooling_sentinter

# Please note that for each model you should train on the fold0~9 dataset and save their top2 ckpt.You can directly use run_fold0~9.py to use the corresponding training set.
```

## Getting Results
Once you get all the ckpt(should be 63 altogether), you need to modify some variables in the test_ensemble_1.sh file (change to your ckpt path), and then run the script file to get intermediate results:
```
./test_ensemble_1.sh
```

In order to help you get test.json results easily, we have saved the intermediate results in the path "test_ensemble_score_5e_tokenmaxpooling_block_63/", you can use them directly

Once you have all the intermediate results and save it to "test_ensemble_score_5e_tokenmaxpooling_block_63", you can run:
```
python results_ensemble_ave_yuzhi.py
```
and you will get the final results here: SemEval/final_results/