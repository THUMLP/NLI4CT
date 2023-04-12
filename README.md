# THiFLY Research at SemEval-2023 Task 7
Codes of the SemEval2023 paper: THiFLY Research at SemEval-2023 Task 7: A Multi-granularity System for CTR-based Textual Entailment and Evidence Retrieval
# Textual Entailment
## Environment
- torch==1.7.1
- torch-scatter==2.0.5
- transformers==4.18.0
- tensorboardX==1.8
- pytorch-pretrained-bert==0.6.2

Please download the necessary pre-trained models from [huggingface](https://huggingface.co/).
## Train & Evaluate
For inference models, you may directly run the command:
 ```
  cd textual_entailment
  FOLD=0
  python xxx.py --do_train --do_eval --fold $FOLD --output_dir xxx_${FOLD}
 ```
 Here xxx can be {512_bi_bi_mul, 512_tf_bi_cl, 1024_tf_bi_mul, scifive}. The argument $FOLD can be varied from 1 to 10.
To evaluate an model, you may run
```
  FOLD=0
  python xxx.py --do_eval --fold $FOLD --output_dir xxx_${FOLD} --load_dir xxx_${FOLD}/saved_model
```
For joint inference network, run
```
  python run_joint_inference.py --do_train --do_eval
```
and
```
  python run_joint_inference.py --do_eval --load_dir outputs_biolinkbert_joint_inference/saved_model
```
for training and evaluating.
## Getting Results
Once you get all the results from the inference models and the joint inference network, run
```
python ensemble_avg.py
```
to ensemble the models and get the final results.



# Evidence Retrieval
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
```
Please note that for each model you should train on the fold0~9 dataset and save their top2 ckpt. You can directly run "run_fold0~9.py" to use the corresponding training set.

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
