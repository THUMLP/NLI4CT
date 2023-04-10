"""
    乱序，保证batch内标签平衡，无taskB
"""
from __future__ import absolute_import, division, print_function
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import random
import sys
import re
import json
import numpy as np
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from pytorch_pretrained_bert.file_utils import  WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from tensorboardX import SummaryWriter
import logging
from models import model_tf_bi_batch
import torch.backends.cudnn as cudnn
logger = logging.getLogger(__name__)

entity_linking_pattern = re.compile('\u2726.*?\u25C6-*[0-9]+,(-*[0-9]+)\u2726')
fact_pattern = re.compile('\u2726(.*?)\u25C6-*[0-9]+,-*[0-9]+\u2726')
unk_pattern = re.compile('\u2726([^\u2726]+)\u25C6-1,-1\u2726')
TSV_DELIM = "\t"
TBL_DELIM = ";"



def parse_fact(fact):
    fact = re.sub(unk_pattern, '[UNK]', fact) # optional
    chunks = re.split(fact_pattern, fact)
    output = ' '.join([x.strip() for x in chunks if len(x.strip()) > 0])
    return output

def truncate_multi_line_data(lines: list, max_length: int):
        # lines中的每一个元素都是一个数组
        lengths = [len(l) for l in lines]
        total_length = np.sum(lengths)

        while total_length > max_length:
            max_idx = np.argmax(lengths)
            lines[max_idx].pop(-1)
            lengths[max_idx] -= 1
            total_length -= 1

        return lines, lengths

def truncate_multi_line_center(lines: list, max_length: int):
    """
    从句子的中间删除
    """
    # lines中的每一个元素都是一个数组
    lengths = [len(l) for l in lines]
    total_length = np.sum(lengths)

    while total_length > max_length:
        max_idx = np.argmax(lengths)
        mid = lengths[max_idx] // 2
        lines[max_idx].pop(mid)  # 每次从中间开始删除
        lengths[max_idx] -= 1
        total_length -= 1

    return lines, lengths

def get_evidence(evidence, evidence_index, block_size):
    tmp, all_evidence = [], []
    tmp_labels, all_labels = [], []
    tmp_ids, all_ids = [], []
    label_map = {}
    for idx in range(len(evidence)):
        label_map[idx] = []
    
    flag = True
    assert len(evidence) == len(evidence_index)
    for i, line in enumerate(evidence):
        if not line.startswith("  ") and flag:
            flag = False
            if len(tmp) > 0:
                all_evidence.append(tmp)
                all_labels.append(tmp_labels)
                all_ids.append(tmp_ids)
            tmp = [line]
            tmp_labels = [evidence_index[i]]
            tmp_ids = [i]
        else:
            tmp.append(line)
            tmp_labels.append(evidence_index[i])
            tmp_ids.append(i)
            if line.startswith("  "):
                flag = True
    all_evidence.append(tmp)
    all_labels.append(tmp_labels)
    all_ids.append(tmp_ids)
    outs, outs_labels = [], []
    cnt, idx = 0, 1
    tmp, tmp_labels = [], []
    flag = True
    for j, block in enumerate(all_evidence):
        for i, line in enumerate(block):
            if flag and i > 0:
                tmp.append('[SEP] '+block[0])
                label_map[all_ids[j][0]].append(idx)
                idx += 1
                tmp_labels.append(all_labels[j][0])
                flag = False
            tmp.append('[SEP] '+line)
            tmp_labels.append(all_labels[j][i])
            label_map[all_ids[j][i]].append(idx)
            idx += 1
            if i == 0:
                flag = False
            if i > 0:
                cnt += 1
            if cnt == block_size:
                cnt = 0
                outs.append(tmp)
                outs_labels.append(tmp_labels)
                tmp = []
                tmp_labels = []
                flag = True
                idx += 1
    if len(tmp) > 0:
        outs.append(tmp)
        outs_labels.append(tmp_labels)
    return outs, outs_labels, label_map

class InputExample(object):
    def __init__(self, statement_id, text_a, text_b, label, evidence_labels):
        '''
        Args:
            guid:   unique id
            text_a: statement
            text_b: table_str
            label:  positive / negative
        '''
        self.statement_id = statement_id
        self.statement = text_a
        self.evidence_strs = text_b
        self.label = label
        self.evidence_labels = evidence_labels


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, evidence_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.evidence_labels = evidence_labels


class LpaProcessor(object):
    def get_examples(self, data_dir, dataset=None, fold=0):
        logger.info('Get examples from: {}.json'.format(dataset))
        if 'test' in dataset:
            return self._create_examples(self._read_tsv("{}.json".format(dataset)))
        else:
            return self._create_examples(self._read_tsv(os.path.join("new_CV","{}_{}.json".format(dataset,fold))))

    def get_labels(self):
        return [0, 1], len([0, 1])
    def load_evidence(self, CTR_id):
        with open(os.path.join('CT_json',CTR_id+'.json')) as f:
            table_data = json.load(f)
        evidence = []
        for name in ['Adverse Events', 'Eligibility', 'Results', 'Intervention']:
            tmp = []
            for line in table_data[name]:
                tmp.append(line)
            evidence.append(tmp)
        return evidence

    def convert_evidence_label(self, evidence_label, evidence_len):
        new_label = []
        for i in range(evidence_len):
            if i in evidence_label:
                new_label.append(1)
            else:
                new_label.append(0)
        return new_label

    def _read_tsv(cls, input_file):
        lines = []
        section_names = ['Adverse Events', 'Eligibility', 'Results', 'Intervention']
        with open(input_file, 'r') as f:
            data = json.load(f)
        for key, one in data.items():
            statement_id = key
            prob_type = one['Type']
            primary_id = one['Primary_id']
            if 'test' in input_file:
                Primary_evidence_index = [0]
            else:
                Primary_evidence_index = one['Primary_evidence_index']

            section_id = one['Section_id']
            evidence = cls.load_evidence(primary_id)
            evidence = evidence[section_names.index(section_id)]
            evidence_list = []
            Primary_evidence_index = cls.convert_evidence_label(Primary_evidence_index, len(evidence))
            real_evidence_label = Primary_evidence_index
            evidence_list.append(evidence)
            if prob_type == 'Comparison':
                secondary_id = one['Secondary_id']
                if 'test' in input_file:
                    Secondary_evidence_index = [0]
                else:
                    Secondary_evidence_index = one['Secondary_evidence_index']
                second_evidence = cls.load_evidence(secondary_id)
                second_evidence = second_evidence[section_names.index(section_id)]
                Secondary_evidence_index = cls.convert_evidence_label(Secondary_evidence_index, len(second_evidence))
                real_evidence_label += Secondary_evidence_index
                evidence_list.append(second_evidence)
            
            statement = one['Statement']
            if 'test' in input_file:
                label = 0
            else:
                label = 1 if one['Label'] == 'Entailment' else 0
            lines.append([statement_id,statement, evidence_list, label, real_evidence_label])
        return lines
        

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            statement_id = line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            evidence_label_list = line[4]
            examples.append((InputExample(statement_id=statement_id, text_a=text_a, text_b=text_b, label=label, evidence_labels=evidence_label_list)))
        return examples

def truncate_multi_line_data(lines: list, max_length: int):
        # lines中的每一个元素都是一个数组
    lengths = [len(l) for l in lines]
    total_length = np.sum(lengths)

    while total_length > max_length:
        max_idx = np.argmax(lengths)
        lines[max_idx].pop(-1)
        lengths[max_idx] -= 1
        total_length -= 1

    return lines, lengths

def truncate_multi_line_data_average(lines: list, max_length: int):
        # lines中的每一个元素都是一个数组
    lengths = [len(l) for l in lines]
    truncline = max(lengths)-1
    total_length = np.sum(lengths)
    while total_length > max_length:
        for i,l in enumerate(lengths):
            if l > truncline:
                lines[i].pop(-1)
                lengths[i] -= 1
                total_length -= 1
        truncline -= 1

    return lines, lengths

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, phase):
    label_map = {label: i for i, label in enumerate(label_list)}

    row_level_features = []

    for (ex_index, example) in enumerate(tqdm(examples, desc="convert to features")):

        label_id = label_map[example.label]
        statement_tokens = tokenizer.tokenize(example.statement)
        tmp_features = []
        evidence_strs = example.evidence_strs
        evidence_labels = example.evidence_labels
        evidence_tokens = []
        evidence_tokens.append([tokenizer.tokenize('primary trial: ')])
        for text in evidence_strs[0]:
            tk = tokenizer.tokenize(text)
            if len(tk) == 0:
                tk = ['[PAD]']
            evidence_tokens[0].append(tk)
        if len(evidence_strs) > 1: # comparison
            evidence_tokens.append([tokenizer.tokenize('secondary trial: ')])
            for text in evidence_strs[1]:
                tk = tokenizer.tokenize(text)
                if len(tk) == 0:
                    tk = ['[PAD]']
                evidence_tokens[1].append(tk)
        primary_num = len(evidence_tokens[0])
        all_evidence_tokens = evidence_tokens[0] if len(evidence_strs) == 1 else evidence_tokens[0]+evidence_tokens[1]
        trunc_all_evidence_tokens, evidence_length = truncate_multi_line_data(all_evidence_tokens,max_seq_length-len(statement_tokens)-3)
        tokens = ['[CLS]']
        section_ids = [0]
        tokens += statement_tokens
        section_ids += [1] * len(statement_tokens)
        tokens += ['[SEP]']
        section_ids += [0]
        for i in range(primary_num):
            assert evidence_length[i] > 0
            tokens += trunc_all_evidence_tokens[i]
            if i == 0:
                section_ids += [0]*evidence_length[i]
            else:
                section_ids += [i+1]*evidence_length[i]
        if len(evidence_strs) > 1:
            for i in range(len(trunc_all_evidence_tokens)-primary_num):
                assert evidence_length[i+primary_num] > 0
                tokens += trunc_all_evidence_tokens[i+primary_num]
                if i == 0:
                    section_ids += [0]*evidence_length[i+primary_num]
                else:
                    section_ids += [i+primary_num]*evidence_length[i+primary_num]
        tokens += ['[SEP]']
        section_ids += [0]
        att_mask = [1]*len(tokens)
        pad_length = max_seq_length - len(tokens)
        if pad_length > 0:
            tokens += ['[PAD]']*pad_length
            att_mask += [0]*pad_length
            section_ids += [0]*pad_length
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        row_level_features.append(InputFeatures(input_ids=input_ids, 
                                                input_mask=att_mask, 
                                                segment_ids=section_ids, 
                                                label_id=example.label,
                                                evidence_labels=example.evidence_labels))
    return row_level_features


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
    f1 = 2*precision*recall/(precision+recall)
    return TP, TN, FN, FP, precision, recall, success, fail, acc, f1


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
              "precision": precision, "recall": recall, "F1":f1, "success": success, "fail": fail, "acc": acc}

    return result


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # trunc estimate part end

class SupConLossPLMS(torch.nn.Module):
    """Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning: https://arxiv.org/abs/2011.01403
    """
    def __init__(self, device, temperature=0.05):
        super(SupConLossPLMS, self).__init__()
        self.tem = temperature
        self.device = device

    def forward(self, batch_emb, labels=None):
        labels = labels.view(-1, 1)
        batch_size = batch_emb.shape[0]
        mask = torch.eq(labels, labels.T).float()
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        # compute logits
        dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), self.tem)
        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)  # _返回索引
        logits = dot_contrast - logits_max.detach()
        # 索引应该保证设备相同
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_sum = mask.sum(1)
        # 防止出现NAN
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
        return mean_log_prob_pos.mean()

def get_dataLoader(args, processor, tokenizer, phase=None):
    dataset_dict = {"train": args.train_set, "dev": args.dev_set, "std_test": args.std_test_set,
                    "complex_test": args.complex_test_set,
                    "small_test": args.small_test_set, "simple_test": args.simple_test_set}
    label_list, _ = processor.get_labels()
    examples = processor.get_examples(args.data_dir, dataset_dict[phase], args.fold)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, phase)

    batch_size = args.train_batch_size if phase == "train" else args.eval_batch_size
    epoch_num = args.num_train_epochs if phase == "train" else 1
    num_optimization_steps = int(len(examples) / args.gradient_accumulation_steps) * epoch_num
    logger.info("Examples#: {}, Batch size: {}".format(len(examples), batch_size * args.gradient_accumulation_steps))
    logger.info("Total num of steps#: {}, Total num of epoch#: {}".format(num_optimization_steps, epoch_num))

    
    all_input_ids = [torch.tensor(f.input_ids, dtype=torch.long) for f in features]
    all_input_mask = [torch.tensor(f.input_mask, dtype=torch.long) for f in features]
    all_segment_ids = [torch.tensor(f.segment_ids, dtype=torch.long) for f in features]
    all_evidence_labels= [torch.tensor(f.evidence_labels, dtype=torch.long) for f in features]
    all_label_ids = [torch.tensor(f.label_id, dtype=torch.long) for f in features]
    
    
    #all_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = []
    index = [i for i in range(len(all_input_ids))]
    if phase == 'train':
        ent_ind, ref_ind = [], []
        for i in index:
            if features[i].label_id == 1:
                ent_ind.append(i)
            else:
                ref_ind.append(i)
        random.shuffle(ent_ind)
        random.shuffle(ref_ind)

        new_index = []
        for ei,ri in zip(ent_ind, ref_ind):
            new_index += [ei, ri]
        index = new_index
        new_examples = []
        for i in range(len(index)):
            new_examples.append(examples[index[i]])
        examples = new_examples
    if phase == 'train':
        cnt = 0
        tmp = [[],[],[],[],[]]
        for idx in index:
            for i,one in enumerate([all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_evidence_labels]):
                tmp[i].append(one[idx])
            cnt += 1
            if cnt == args.train_batch_size:
                cnt = 0
                dataloader.append(tmp)
                tmp = [[],[],[],[],[]]
        if len(tmp[0]) > 0:
            dataloader.append(tmp)
    else:
        for idx in index:
            dataloader.append((all_input_ids[idx], all_input_mask[idx], all_segment_ids[idx], all_label_ids[idx], all_evidence_labels[idx]))

    return dataloader, num_optimization_steps, examples


def save_model(model_to_save, tokenizer, step=-1, res=0, name='saved_model'):
    save_model_dir = os.path.join(args.output_dir, name)
    mkdir(save_model_dir)
    output_model_file = os.path.join(save_model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_model_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    #model_to_save.bert_1.config.to_json_file(output_config_file)
    #tokenizer.save_vocabulary(save_model_dir)




def run_train(device, processor, tokenizer,model, writer, phase="train"):
    logger.info("\n************ Start Training *************")

    tr_dataloader, tr_num_steps, tr_examples = get_dataLoader(args, processor, tokenizer, phase="train")

    model.train()

    loss_fct = torch.nn.BCEWithLogitsLoss(reduction="sum")
    conloss_fct = SupConLossPLMS(device, args.temperature)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = \
        [{'params': [p for n, p in param_optimizer if 'pred_model' not in n], 'weight_decay': 0.01, 'lr':args.lstm_lr},
        {'params': [p for n, p in param_optimizer if ('pred_model' in n) and (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
         {'params': [p for n, p in param_optimizer if ('pred_model' in n) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=tr_num_steps)
    optimizer.zero_grad()

    global_step = 0
    best_F1, best_evi_F1 = 0.0, 0.0
    _, num_labels = processor.get_labels()
    n_gpu = torch.cuda.device_count()
    all_label, all_pred = [],[]
    all_loss = 0.0
    all_evidence_loss = 0.0
    num_steps = 0
    scaler = GradScaler()
    

    for ep in trange(args.num_train_epochs, desc="Training"):
        if ep > 0:
            tr_dataloader, tr_num_steps, tr_examples = get_dataLoader(args, processor, tokenizer, phase="train")
        for step, batch in enumerate(tqdm(tr_dataloader)):
            evidence_labels = [t.to(device) for t in batch[-1]]
            if len(batch[0]) > 1:
                new_batch = []
                for i in range(len(batch)-1):
                    new_batch.append(torch.stack(batch[i], dim=0))
                batch = new_batch
            else:
                batch = list(batch)[:-1]


            batch = tuple(t.to(device) for t in batch)
            
            input_ids, input_mask, segment_ids, label_ids = batch
            inputs = [input_ids, input_mask, segment_ids]
            prob, evidence_logits, nli_repre = model(inputs)

            actual_prob = torch.exp(prob)
            all_label += label_ids.detach().cpu().tolist()
            all_pred += torch.argmax(actual_prob, dim=1).detach().cpu().tolist()
            loss = F.nll_loss(prob, label_ids.view(-1), reduction='mean')
            contra_loss = conloss_fct(nli_repre, label_ids)
            # evidence_labels_float = evidence_labels.float()
            loss_2 = []
            for i,one in enumerate(evidence_labels):
                tmp_loss_2 = loss_fct(evidence_logits[i,:one.size(0)], one.float())
                loss_2.append(tmp_loss_2)
                all_evidence_loss += float(tmp_loss_2/one.size(0))
            loss_2_tensor = torch.stack(loss_2, dim=0).mean()
            all_loss += float(loss)
            
            loss = (1 - args.loss_1_rate) * contra_loss + args.loss_1_rate * loss
            
            num_steps += 1
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            writer.add_scalar('{}/loss'.format(phase), loss, global_step)

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:  # optimizer
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            model.eval()
            torch.set_grad_enabled(False)

            if args.do_eval and step % (args.period*args.gradient_accumulation_steps) == 0:
                model_to_save = model.module if hasattr(model, 'module') else model

                dev_F1, dev_evi_F1 = run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=True,
                                   phase="dev")

                save_detailed_res = False
                if dev_F1 > best_F1:
                    save_detailed_res = True  # save std_test results
                    best_F1 = dev_F1
                    logger.info(">> Save model. Best F1: {:.4}. Epoch {}".format(best_F1, ep))
                    save_model(model_to_save, tokenizer, step=int(global_step / 1000), res=dev_F1, name='saved_model')  # save model
                    logger.info(">> Now the best F1 is {:.4}\n".format(dev_F1))
                    #run_eval(device, processor, tokenizer, model, writer, global_step, save_detailed_res,
                             #tensorboard=True, phase="std_test")
                if dev_evi_F1 > best_evi_F1:
                    save_detailed_res = True  # save std_test results
                    best_evi_F1 = dev_evi_F1
                    logger.info(">> Save model. Best evi F1: {:.4}. Epoch {}".format(best_evi_F1, ep))
                    save_model(model_to_save, tokenizer, step=int(global_step / 1000), res=dev_F1, name='saved_evi_model')  # save model
                    logger.info(">> Now the best evi F1 is {:.4}\n".format(dev_evi_F1))
                    #run_eval(device, processor, tokenizer, model, writer, global_step, save_detailed_res,
                             #tensorboard=True, phase="std_test")
                

            model.train()
            torch.set_grad_enabled(True)
        all_loss/=num_steps
        all_evidence_loss /= num_steps
        result = compute_metrics(np.asarray(all_pred), np.asarray(all_label))
        result['{}_loss'.format(phase)] = all_loss
        result['{}_evidence_loss'.format(phase)] = all_evidence_loss
        result['epoch'] = ep
        logger.info(result)
        all_label = []
        all_pred = []
        all_loss = 0.0
        num_steps = 0

    return global_step


def run_eval(device, processor, tokenizer, model, writer, global_step, save_detailed_res=False, tensorboard=False,
             phase=None):
    sys.stdout.flush()
    logger.info("\n************ Start {} *************".format(phase))

    model.eval()

    loss_fct = torch.nn.BCEWithLogitsLoss(reduction="sum")

    dataloader, num_steps, examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    eval_loss = 0.0
    evidence_loss = 0.0
    num_steps = 0
    preds = []
    all_labels = []
    mapping = []
    all_evidence_preds, all_evidence_labels = [],[]
    _, num_labels = processor.get_labels()
    class_probs, select_probs = [],[]
    attention_data = {}
    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, evidence_labels = batch
        num_steps += 1

        with torch.no_grad():
            
            start = step
            end = start + 1
            batch_range = list(range(start, end))
            input_ids = input_ids.view(1,-1)
            input_mask = input_mask.view(1,-1)
            segment_ids = segment_ids.view(1,-1)
            inputs = [input_ids, input_mask, segment_ids]
            
            prob, evidence_logits, nli_repre = model(inputs)
            tmp_loss = F.nll_loss(prob, label_ids.view(-1))
            tmp_loss_2 = loss_fct(evidence_logits.squeeze(0), evidence_labels.float())
            eval_loss += tmp_loss.mean().item()
            evidence_loss += tmp_loss_2.mean().item() / evidence_logits.squeeze(0).size(0)
            logits_sigmoid = torch.exp(prob)
            evidence_logits_cpu = F.sigmoid(evidence_logits.squeeze(0)).cpu()
            # all_evidence_preds += torch.argmax(evidence_logits, dim=-1).tolist()
            tmp_evidence_preds = (evidence_logits_cpu > 0.5).long().tolist()
            all_evidence_preds += tmp_evidence_preds
            all_evidence_labels += evidence_labels.tolist()
            if len(preds) == 0:
                preds.append(logits_sigmoid.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits_sigmoid.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()

            

            statement_id = [examples[i].statement_id for i in batch_range]
            t_a = [examples[i].statement for i in batch_range]
            t_b = []
            for i in batch_range:
                evi_strs = []
                for one in examples[i].evidence_strs:
                    evi_strs += one
                t_b.append('\u2726'.join(evi_strs))
            # t_b = ['\u2726'.join(examples[i].evidence_strs) for i in batch_range]
            similarity = logits_sigmoid.detach().cpu().numpy().tolist()
            labels = label_ids.detach().cpu().numpy().tolist()

            assert len(t_a) == len(t_b) == 1
            all_labels.append(labels)
            mapping.append([statement_id[0], str(t_a[0]), str(t_b[0]), str(labels), str(similarity[0][1]), str(tmp_evidence_preds)])
            

    eval_loss /= num_steps
    evidence_loss /= num_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    # json.dump([class_probs,select_probs], open('./{}_block_extra.json'.format(phase),'w'))
    result = compute_metrics(preds, np.asarray(all_labels))
    new_result = compute_metrics(np.asarray(all_evidence_preds),np.asarray(all_evidence_labels))
    result['{}_loss'.format(phase)] = eval_loss
    result['{}_evidence_loss'.format(phase)] = evidence_loss
    result['global_step'] = global_step
    result['{}_evidence_F1'.format(phase)] = new_result['F1']
    logger.info(result)
    #logger.info(np.asarray(all_labels))
    json.dump(mapping, open('./{}_result_512_tf_bi_cl_{}.json'.format(phase,args.fold),'w', encoding='utf8'))
    if tensorboard and writer is not None:
        for key in sorted(result.keys()):
            writer.add_scalar('{}/{}'.format(phase, key), result[key], global_step)

    model.train()

    return result['F1'], new_result['F1']


def main():
    mkdir(args.output_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))
    cache_dir = args.cache_dir

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True

    save_code_log_path = args.output_dir

    logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %H:%M', level=logging.INFO,
                        handlers=[logging.FileHandler("{0}/{1}.log".format(save_code_log_path, 'output')),
                                  logging.StreamHandler()])
    logger.info(args)
    logger.info("Command is: %s" % ' '.join(sys.argv))
    logger.info("Device: {}, n_GPU: {}".format(device, n_gpu))
    logger.info("Datasets are loaded from {}\nOutputs will be saved to {}\n".format(args.data_dir, args.output_dir))

    processor = LpaProcessor()

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    load_dir = args.load_dir if args.load_dir else args.bert_model
    logger.info('Model is loaded from %s' % load_dir)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    bert = BertModel.from_pretrained(args.bert_model)
    bert.gradient_checkpointing_enable()
    model = model_tf_bi_batch(bert,os.path.join(args.bert_model,'config.json'),num_layers=args.tf_num_layers, pool=args.pool,num_head=args.num_head)
    if args.load_dir:
        model.load_state_dict(torch.load(load_dir+'/pytorch_model.bin'))
        print('parameters loaded successfully.')
    model.to(device)

    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model,device_ids=[0, 1])

    if args.do_train:
        run_train(device, processor, tokenizer, model, writer, phase="train")

    if args.do_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="dev")
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="std_test")

    if args.do_train_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="train")

    if args.do_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, save_detailed_res=True, tensorboard=False,
                 phase="std_test")

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
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_train_eval", action='store_true')
    parser.add_argument("--do_simple_test", action='store_true')
    parser.add_argument("--do_complex_test", action='store_true')
    parser.add_argument("--do_small_test", action='store_true')
    parser.add_argument("--load_dir", help="load model checkpoints")
    parser.add_argument("--data_dir", help="path to data", default='../processed_datasets_fulltable_horizontal_h2o1/tsv_data_horizontal_block')
    parser.add_argument("--train_set", default="train")
    parser.add_argument("--dev_set", default="dev")
    parser.add_argument("--std_test_set", default="test")
    parser.add_argument("--small_test_set", default="small_test")
    parser.add_argument("--complex_test_set", default="complex_test")
    parser.add_argument("--simple_test_set", default="simple_test")
    parser.add_argument("--output_dir", default='outputs_biolinkbert_multitask_sent_transformer_1_batch_cse_rand')
    parser.add_argument("--cache_dir", default="/home/zhouyx/tapas-master/tapas-master/SemEval_v1.2/code/roberta", type=str, help="store downloaded pre-trained models")
    parser.add_argument('--period', type=int, default=1000)
    parser.add_argument("--bert_model", default="biolinkbert-large-mnli-snli", type=str,
                        help="list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, "
                             "bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--task_name", default="LPA", type=str)
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--overlap", default=0, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--max_train_row_num", default=20, type=int)
    parser.add_argument("--num_head", default=16, type=int)
    parser.add_argument("--loss_1_rate", default=0.5, type=float)
    parser.add_argument("--loss_2_rate", default=0.01, type=float)
    parser.add_argument("--temperature", default=0.3, type=float)
    parser.add_argument("--pool", default='max', type=str)
    parser.add_argument("--train_batch_size", default=32)
    parser.add_argument("--eval_batch_size", default=32)
    parser.add_argument("--tf_num_layers", default=1, type=int)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--lstm_lr", default=1e-4, type=float, help="The initial learning rate for lstm.")
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()
    main()


    # replicate: [52] /home/17xy44/repos/Table-Fact-Checking/nf_code/LPA_outputs/bert-0406-185424/saved_model_step47k_res0.677
    #            [240] /home/17xy44/repos/Table-Fact-Checking/events_compare/bert-0406-185424/saved_model_step47k_res0.677
    # data_dir: ../../preprocessed_data_program
