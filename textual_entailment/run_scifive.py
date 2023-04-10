from __future__ import absolute_import, division, print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import logging
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from transformers import (  # use AdamW is a standard practice for transformer
    Adafactor, T5ForConditionalGeneration, T5Tokenizer,
    get_linear_schedule_with_warmup)

logger = logging.getLogger(__name__)

entity_linking_pattern = re.compile('\u2726.*?\u25C6-*[0-9]+,(-*[0-9]+)\u2726')
fact_pattern = re.compile('\u2726(.*?)\u25C6-*[0-9]+,-*[0-9]+\u2726')
unk_pattern = re.compile('\u2726([^\u2726]+)\u25C6-1,-1\u2726')
TSV_DELIM = "\t"
TBL_DELIM = ";"



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        '''
        Args:
            guid:   unique id
            text_a: statement
            text_b: table_str
            label:  positive / negative
        '''
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class LpaProcessor(object):
    def get_examples(self, dataset=None, fold=0):
        if 'test' in dataset:
            logger.info('Get examples from: {}'.format(os.path.join("{}.json".format(dataset))))
            return self._create_examples(self._read_tsv(os.path.join("{}.json".format(dataset)), dataset))
        else:
            logger.info('Get examples from: {}'.format(os.path.join("new_CV","{}_{}.json".format(dataset,fold))))
            return self._create_examples(self._read_tsv(os.path.join("new_CV","{}_{}.json".format(dataset,fold)), dataset))

    def load_evidence(self, CTR_id, name):
        with open(os.path.join('CT_json',CTR_id+'.json')) as f:
            table_data = json.load(f)
        return table_data[name]

    def _read_tsv(cls, input_file, phase):
        lines = []
        with open(input_file, 'r') as f:
            data = json.load(f)
        for key, one in data.items():
            prob_type = one['Type']
            primary_id = one['Primary_id']
            section_id = one['Section_id']
            evidence = cls.load_evidence(primary_id, section_id)
            evidence = ['primary trial:'] + evidence
            if prob_type == 'Comparison':
                secondary_id = one['Secondary_id']
                second_evidence = cls.load_evidence(secondary_id, section_id)
                evidence += ['secondary trial:']+second_evidence
            statement_id = key
            evidence = ' '.join(evidence)
            statement = one['Statement']
            if phase == 'test':
                label = 0
            else:
                label = 1 if one['Label'] == 'Entailment' else 0
            lines.append([statement_id,statement, evidence, label])
        return lines
        

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            statement_id = line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append((InputExample(guid=statement_id, text_a=text_a, text_b=text_b, label=label)))
        return examples
    def _convert_examples_to_inputs(self, examples):
        source_list, target_list = [],[]
        for example in examples:
            source_text = 'nli hypothesis: '+example.text_a + 'premise: '+ example.text_b
            target = 'entailment' if example.label == 1 else 'contradiction'
            source_list.append(source_text)
            target_list.append(target)
        return source_list, target_list

def eval_1(preds, labels):
    TP = ((preds == 1) & (labels == 1)).sum()
    FN = ((preds == 0) & (labels == 0)).sum()
    TN = ((preds == 0) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + TN + 0.001)
    success = TP + FN
    fail = TN + FP
    acc = success / (success + fail + 0.001)
    f1 = 2*recall*precision/(recall+precision)
    return TP, TN, FN, FP, precision, recall, success, fail, acc,f1


def eval_2(mapping):
    success = 0
    fail = 0
    for idx in mapping.keys():
        similarity, prog_label, fact_label, gold_label = mapping[idx]
        if prog_label == fact_label:
            success += 1
        else:
            fail += 1
    acc = success / (success + fail + 0.001)

    return success, fail, acc


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    TP, TN, FN, FP, precision, recall, success, fail, acc, f1 = eval_1(preds, labels)
    result = {"TP": TP, "TN": TN, "FN": FN, "FP": FP,
              "precision": precision, "recall": recall, "success": success, "fail": fail, "acc": acc, "f1":f1}

    return result


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataLoader(args, processor, tokenizer, phase=None):
    dataset_dict = {"train": args.train_set, "dev": args.dev_set, "std_test": args.std_test_set,
                    "complex_test": args.complex_test_set,
                    "small_test": args.small_test_set, "simple_test": args.simple_test_set}


    dataset = processor.get_examples(dataset_dict[phase], args.fold)
    # features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    source_list, target_list = processor._convert_examples_to_inputs(dataset)
    encoding = tokenizer(
    source_list,
    padding="longest",
    max_length=args.max_source_length,
    truncation=True,
    return_tensors="pt",
    )
    target_encoding = tokenizer(
    target_list,
    padding="longest",
    max_length=args.max_target_length,
    truncation=True,
    return_tensors="pt",
    )
    batch_size = args.train_batch_size if phase == "train" else args.eval_batch_size
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    all_data = TensorDataset(input_ids, attention_mask, labels)
    sampler = RandomSampler(all_data) if phase == "train" else SequentialSampler(all_data)
    dataloader = DataLoader(all_data, sampler=sampler, batch_size=batch_size)
        
    return dataloader, dataset


def save_model(model_to_save, tokenizer, step=-1, res=0):
    save_model_dir = os.path.join(args.output_dir, 'saved_model')
    mkdir(save_model_dir)
    output_model_file = os.path.join(save_model_dir, WEIGHTS_NAME)
    # output_config_file = os.path.join(save_model_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    # model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(save_model_dir)


def run_train(device, processor, tokenizer, model, writer, phase="train"):
    logger.info("\n************ Start Training *************")

    tr_dataloader, tr_examples = get_dataLoader(args, processor, tokenizer, phase="train")

    model.train()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters1 = \
        [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer_grouped_parameters2 = [{'params': [p for name, p in model.template.named_parameters() if 'raw_embedding' not in name]}]

    optimizer1 = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
    # optimizer2 = torch.optim.AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr)
    tot_step = int(np.ceil(len(tr_examples)/args.train_batch_size)*args.num_train_epochs)
    optimizer1.zero_grad()
    # optimizer2.zero_grad()
    scheduler = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps =args.warmup_step_prompt, num_training_steps = tot_step)

    global_step = 0
    best_f1 = 0.0
    # _, num_labels = processor.get_labels()
    n_gpu = torch.cuda.device_count()
    all_label, all_pred = [],[]
    all_loss = 0.0
    num_steps = 0
    for ep in trange(args.num_train_epochs, desc="Training"):

        for step, batch in enumerate(tqdm(tr_dataloader)):
            # print(batch['loss_ids'].max(dim=1).values.tolist())
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            results = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


            loss = results.loss
            #print(loss.item())
            # all_label += labels.tolist()
            # all_pred += torch.argmax(logits, dim=1).tolist()
            all_loss += loss.item()
            if n_gpu > 1:
                loss = loss.mean()
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            writer.add_scalar('{}/loss'.format(phase), loss, global_step)

            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:  # optimizer
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer1.step()
                optimizer1.zero_grad()
                scheduler.step()
                
                global_step += 1
            num_steps += 1
            if args.do_eval and step % (args.period*args.gradient_accumulation_steps) == 0:
                model.eval()
                torch.set_grad_enabled(False)

                # if args.do_eval and (((step + 1) % args.gradient_accumulation_steps == 0 and global_step % args.period == 0) or (ep==0 and step==0)):
                model_to_save = model.module if hasattr(model, 'module') else model

                dev_f1 = run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=True,
                                    phase="dev")

                save_detailed_res = False
                if dev_f1 > best_f1:
                    save_detailed_res = True  # save std_test results
                    best_f1 = dev_f1
                    logger.info(">> Save model. Best f1: {:.4}. Epoch {}".format(best_f1, ep))
                    save_model(model_to_save, tokenizer, step=int(global_step / 1000), res=dev_f1)  # save model
                    logger.info(">> Now the best f1 is {:.4}\n".format(dev_f1))
                    #run_eval(device, processor, tokenizer, model, writer, global_step, save_detailed_res,
                                #tensorboard=True, phase="std_test")

                model.train()
                torch.set_grad_enabled(True)
        all_loss/=num_steps
        # result = compute_metrics(np.asarray(all_pred), np.asarray(all_label))
        result = {}
        result['{}_loss'.format(phase)] = all_loss
        result['epoch'] = ep
        logger.info(result)
        all_label = []
        all_pred = []
        all_loss = 0.0
        num_steps = 0

    return global_step

def cal_prob(target_text, input_ids, attention_mask, model, tokenizer, device):

    #将target_text转换为t5模型的输出格式
    labels = tokenizer.encode(target_text, return_tensors="pt", padding=True).to(device) # 由LabeLs生成decoder input ids，需要在前面使得长度与Labels相同
    decoder_input_ids = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=-1).to(device) # 计算生成text的概率
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_input_ids=decoder_input_ids)
    loss = outputs[0]
    text_prob=torch.exp(-loss)**(len(target_text))
    return text_prob

def run_eval(device, processor, tokenizer, model, writer, global_step, save_detailed_res=False, tensorboard=False,
             phase=None):
    sys.stdout.flush()
    logger.info("\n************ Start {} *************".format(phase))

    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    dataloader, examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    
    eval_loss = 0.0
    num_steps = 0
    preds = []
    all_labels = []
    mapping = []
    # _, num_labels = processor.get_labels()
    #output_logits = []
    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        # batch = batch.cuda()
        num_steps += 1

        with torch.no_grad():

            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            # tmp_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            ent_prob = cal_prob('entailment', input_ids, attention_mask, model, tokenizer, device)
            con_prob = cal_prob('contradiction', input_ids, attention_mask, model, tokenizer, device)
            # labels = batch['label']
            # tmp_loss = loss_fct(logits, labels)
            # eval_loss += tmp_loss.mean().item()
            tmp_preds = []
            probability = ent_prob / (ent_prob + con_prob)
            if ent_prob > con_prob:
                preds.append(1)
                tmp_preds.append(1)
            else:
                preds.append(0)
                tmp_preds.append(0)

            start = step * args.eval_batch_size if not args.do_train_eval else step * args.train_batch_size
            end = start + len(labels)
            batch_range = list(range(start, end))

            statement_ids = [examples[i].guid for i in batch_range]
            t_a = [examples[i].text_a for i in batch_range]
            t_b = [examples[i].text_b for i in batch_range]
            labels = [examples[i].label for i in batch_range]

            assert len(t_a) == len(t_b) == len(labels)
            all_labels.extend(labels)
            for i, st_name in enumerate(statement_ids):
                # mapping.append([t_name, str(t_a[i]), str(t_b[i]), str(labels[i]), str(similarity[i][1]),str(loss_list[i])])
                mapping.append([st_name, str(t_a[i]), str(t_b[i]), str(labels[i]), str(tmp_preds[i]), str(float(probability.detach()))])
            
    #output_logits = np.concatenate(output_logits, axis=0)
    #pkl.dump(output_logits, open('table_grappa_train_logits.pkl','wb'))
    # eval_loss /= num_steps
    # preds = np.argmax(preds, axis=1)

    result = compute_metrics(np.asarray(preds), np.asarray(all_labels))
    # result['{}_loss'.format(phase)] = eval_loss
    result['global_step'] = global_step
    logger.info(result)
    #logger.info(np.asarray(all_labels))
    json.dump(mapping, open('./{}_result_scifive_{}.json'.format(phase, args.fold),'w', encoding='utf8'))
    if tensorboard and writer is not None:
        for key in sorted(result.keys()):
            writer.add_scalar('{}/{}'.format(phase, key), result[key], global_step)

    model.train()

    return result['f1']

def main():
    mkdir(args.output_dir)

    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))
    cache_dir = args.cache_dir

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    save_code_log_path = args.output_dir

    logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %H:%M', level=logging.INFO,
                        handlers=[logging.FileHandler("{0}/{1}.log".format(save_code_log_path, 'output')),
                                  logging.StreamHandler()])
    logger.info(args)
    logger.info("Command is: %s" % ' '.join(sys.argv))
    logger.info("Device: {}, n_GPU: {}".format(device, n_gpu))
    logger.info("Datasets are loaded from {}\nOutputs will be saved to {}\n".format(args.data_dir, args.output_dir))

    processor = LpaProcessor()
    tokenizer = T5Tokenizer.from_pretrained(args.bert_model)
    model = T5ForConditionalGeneration.from_pretrained(args.bert_model)

    if args.load_dir:
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'pytorch_model.bin')))
    model.to(device)
    # label_list = processor.get_labels()
    # num_labels = len(label_list)
    # model = RobertaForSequenceClassification.from_pretrained(load_dir, cache_dir=cache_dir, num_labels=num_labels)
    # model.config.gradient_checkpointing = True
    # model.to(device)

    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model,device_ids=[0, 1])

    if args.do_train:
        run_train(device, processor, tokenizer, model, writer, phase="train")

    if args.do_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="dev")
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="std_test")
        
    if args.do_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="std_test")

    if args.do_train_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="train")

    if args.do_complex_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="complex_test")

    if args.do_small_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="small_test")

    if args.do_simple_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="simple_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_train_eval",action='store_true')
    parser.add_argument("--do_simple_test", action='store_true')
    parser.add_argument("--do_complex_test", action='store_true')
    parser.add_argument("--do_small_test", action='store_true')
    parser.add_argument("--load_dir", help="load model checkpoints")
    parser.add_argument("--data_dir", help="path to data", default='../tsv_data_horizontal')
    parser.add_argument("--train_set", default="train")
    parser.add_argument("--dev_set", default="dev")
    parser.add_argument("--std_test_set", default="test")
    parser.add_argument("--small_test_set", default="small_test")
    parser.add_argument("--complex_test_set", default="complex_test")
    parser.add_argument("--simple_test_set", default="simple_test")
    parser.add_argument("--output_dir", default='outputs_scifive_tuning_std_prob_4')
    parser.add_argument("--cache_dir", default="SciFive", type=str, help="store downloaded pre-trained models")
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument("--bert_model", default="SciFive", type=str,
                        help="list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, "
                             "bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--task_name", default="LPA", type=str)
    parser.add_argument("--max_source_length", default=1024)
    parser.add_argument("--max_target_length", default=5)
    parser.add_argument("--train_batch_size", default=1)
    parser.add_argument("--eval_batch_size", default=1)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--prompt_lr", default=0.3, type=float, help="The prompt learning rate for Adam.")
    parser.add_argument("--warmup_step_prompt", default=500, type=int, help="usually num_warmup_steps is 500")
    parser.add_argument("--num_train_epochs", default=100)
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--seed', type=int, default=40, help="random seed")

    args = parser.parse_args()
    main()


    # replicate: [52] /home/17xy44/repos/Table-Fact-Checking/nf_code/LPA_outputs/bert-0406-185424/saved_model_step47k_res0.677
    #            [240] /home/17xy44/repos/Table-Fact-Checking/events_compare/bert-0406-185424/saved_model_step47k_res0.677
    # data_dir: ../../preprocessed_data_program
