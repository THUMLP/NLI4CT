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
