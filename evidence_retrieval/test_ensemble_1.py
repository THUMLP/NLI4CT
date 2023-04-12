# coding: UTF-8
import os
from importlib import import_module
import json
import torch
import torch.nn as nn
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'Sentlevel_Section_new_3', help='the path of the dataset') 
    parser.add_argument('--model_name', type=str, default= 'biolinkbert_new_sentpooling_sentinter', help='the model name')
    parser.add_argument('--ckpt_load', type=str, default= 'fold0_new_5e/saved_dict/biolinkbert.ckpt', help='the path of ckpt')
    parser.add_argument('--output_file', type=str, default= 'test_ensemble/test_ensemble_0.json', help='the path of the output file')

    args = parser.parse_args()
    return args


args = parse_arguments()
print('####'*50)
for k in list(vars(args).keys()):
    print(f'{k}: {vars(args)[k]}')

dataset = args.dataset
# model's name
model_name = args.model_name
# loading ckpt
ckpt_load = args.ckpt_load
premise_path = "SemEval/Training data/CT json/"
CT_files = os.listdir(premise_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k_num = {"Intervention":3, "Eligibility":2, "Results":4, "Adverse Events":2}
test_file = 'SemEval/data/test.json'
output_file = args.output_file
model_name_tokenmaxpooling_beiyong = 'biolinkbert_sent_tokenpooling_sentinter_zs'
model_name_block_beiyong = 'biolinkbert_new_sentpooling_sentinter_block_mzs'

if model_name == 'biolinkbert_sent_tokenpooling_sentinter':
    try:
        x = import_module('models.' + model_name)
        config = x.Config(dataset)
        model = x.Model(config).to(device)
        model.load_state_dict(torch.load(ckpt_load))
        model.eval()
    except:
        x = import_module('models.' + model_name_tokenmaxpooling_beiyong)
        config = x.Config(dataset)
        model = x.Model(config).to(device)
        model.load_state_dict(torch.load(ckpt_load))
        model.eval()
elif model_name == 'biolinkbert_new_sentpooling_sentinter_block':
    try:
        x = import_module('models.' + model_name)
        config = x.Config(dataset)
        model = x.Model(config).to(device)
        model.load_state_dict(torch.load(ckpt_load))
        model.eval()
    except:
        x = import_module('models.' + model_name_block_beiyong)
        config = x.Config(dataset)
        model = x.Model(config).to(device)
        model.load_state_dict(torch.load(ckpt_load))
        model.eval()
else:
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(device)
    model.load_state_dict(torch.load(ckpt_load))
    model.eval()



# <premise json file> section sentences
def get_section_sent(pre_json,section,trial):
    """
    :param pre_json: NCT01669343.json
    :param section: Intervention
    """
    pre_json = premise_path + pre_json
    with open(pre_json, 'r', encoding='utf-8') as f:
        dic = json.load(f)
        temp_string_list = dic[section]
        result = []
        for i,sent in enumerate(temp_string_list):
            if i == 0:
                sent = trial + " " + sent  
            sent = sent.strip()
            sent = sent.replace('\t', '')
            if sent:
                result.append(sent)
            else: 
                result.append('None')
 
    return result


def data_preprocess(key): 
    """
    :param key: data
    """
    per_data_dict = {}
    statement = dic[key]["Statement"]
    section = dic[key]["Section_id"]
    line_list = []
    if dic[key]["Type"] == "Single":
        pre_json = dic[key]["Primary_id"] + ".json"
        trial = "primary trial"
        pri_sent_list = get_section_sent(pre_json, section, trial)
    
        line_list.extend(pri_sent_list)
        line_list.append(statement)
        per_data_dict["statement_id"] = key
        per_data_dict["section"] = section
        per_data_dict["primary_sentence"] = line_list

    else:
        pre_primary_json = dic[key]["Primary_id"] + ".json"
        pre_secondary_json = dic[key]["Secondary_id"] + ".json"
        pri_trial = "primary trial"
        sec_trial = "secondary trial"
        pri_sent_list = get_section_sent(pre_primary_json, section, pri_trial)
        sec_sent_list = get_section_sent(pre_secondary_json, section, sec_trial)   

        line_list.extend(pri_sent_list)
        line_list.append(statement)

        per_data_dict["statement_id"] = key
        per_data_dict["section"] = section
        per_data_dict["primary_sentence"] = line_list


        line_list = []
        line_list.extend(sec_sent_list)
        line_list.append(statement)
        
        per_data_dict["secondary_sentence"] = line_list

    return per_data_dict


def build_dataset(config, per_data_list, pad_size=512):
    contents = []
    statement = per_data_list[-1]
    evi_sent_list = per_data_list[:-1]
    num_evi_sent = len(evi_sent_list)

    token_statement = config.tokenizer.tokenize(statement)

    # cls+state+sep+sen1+sep+sen2+...+sep+senn
    num_sep = num_evi_sent + 1
    num_cls = 1

    aver_len = int((pad_size - num_cls - num_sep - len(token_statement))/num_evi_sent)
    per_data_token = []
    per_data_token.append('[CLS]')
    per_data_token.extend(token_statement)
    for sen in evi_sent_list:
        per_data_token.append('[SEP]')
        if len(config.tokenizer.tokenize(sen)) <= aver_len:   
            per_data_token.extend(config.tokenizer.tokenize(sen))
        else:
            per_data_token.extend(config.tokenizer.tokenize(sen)[0:aver_len])
    per_data_token.append('[SEP]')

    # token2id
    per_data_token_ids = config.tokenizer.convert_tokens_to_ids(per_data_token)

    # 获取每个句子的区间
    cls_sep_index_list = []
    cls_sep_index_list.append(0)
    cls_sep_index_list.extend([i for i,x in enumerate(per_data_token_ids) if x == 3])   # 添加sep索引

    state_sen_interval_list = []
    for i in range(len(cls_sep_index_list)-1):
        state_sen_interval_list.append((cls_sep_index_list[i]+1, cls_sep_index_list[i+1]-1))

    segment_ids = []
    segment_ids.extend([0]*(len(token_statement)+2))
    for i in range(1,len(state_sen_interval_list)):
        segment_ids.extend([i]*(state_sen_interval_list[i][1]-state_sen_interval_list[i][0]+2))

    seq_len = len(per_data_token_ids)
    mask = []
    if pad_size:
        if len(per_data_token_ids) < pad_size:
            mask = [1] * len(per_data_token_ids) + [0] * (pad_size - len(per_data_token_ids))
            per_data_token_ids += ([0] * (pad_size - len(per_data_token_ids)))
            segment_ids += ([num_evi_sent] * (pad_size - seq_len)) 
        else:
            mask = [1] * pad_size
            seq_len = pad_size
    contents.append((per_data_token_ids, seq_len, mask, segment_ids, state_sen_interval_list)) 
    return contents

def build_dataloader(config, content):
    x = torch.LongTensor([_[0] for _ in content]).to(config.device)
    seq_len = torch.LongTensor([_[1] for _ in content]).to(config.device)
    mask = torch.LongTensor([_[2] for _ in content]).to(config.device)
    seg = torch.LongTensor([_[3] for _ in content]).to(config.device)
    interval = torch.LongTensor([_[4] for _ in content]).to(config.device)

    return (x, seq_len, mask, seg, interval)

result_dic = {}
with open(test_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as writer:
    dic = json.load(f)
    key_list = dic.keys()
    for key in key_list:
        per_data_dict = data_preprocess(key)

        if "primary_sentence" in per_data_dict and "secondary_sentence" not in per_data_dict: # single
            per_data_list = per_data_dict["primary_sentence"]
            primary_content = build_dataset(config, per_data_list, pad_size=512)
            primary_dataloder = build_dataloader(config, primary_content)
            outputs = model(primary_dataloder)
            m = nn.Softmax(dim=1)
            predic = m(outputs.data).cpu().numpy()   

            first_pred_idx = {}
            
            for i in range(len(predic)):
                first_pred_idx[str(i)]=float(predic[i][1])

            result_dic[key] = {"Primary_evidence_index":first_pred_idx}
      
        else:
            per_data_list = per_data_dict["primary_sentence"]
            primary_content = build_dataset(config, per_data_list, pad_size=512)
            primary_dataloder = build_dataloader(config, primary_content)  
            outputs = model(primary_dataloder)
            m = nn.Softmax(dim=1)
            predic = m(outputs.data).cpu().numpy()  

            first_pred_idx = {}
            for i in range(len(predic)):
                first_pred_idx[str(i)]=float(predic[i][1])

            per_data_list = per_data_dict["secondary_sentence"]
            secondary_content = build_dataset(config, per_data_list, pad_size=512)
            secondary_dataloder = build_dataloader(config, secondary_content) 
            outputs = model(secondary_dataloder)
            predic = m(outputs.data).cpu().numpy()
            
            second_pred_idx = {}
            for i in range(len(predic)):
                second_pred_idx[str(i)]=float(predic[i][1])

            result_dic[key] = {"Primary_evidence_index":first_pred_idx, "Secondary_evidence_index":second_pred_idx}
        
    json.dump(result_dic, writer,ensure_ascii=False)
