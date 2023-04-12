# coding: UTF-8
import json
import os

def mean(arr):
    result = 0
    for i in arr:
        result+=i
    return result/len(arr)

test_file_path = 'test_ensemble_score_5e_tokenmaxpooling_block_63/'
test_file_list = os.listdir(test_file_path)
output_file = 'SemEval/final_results/results.json'

with open(test_file_path+test_file_list[0], 'r', encoding='utf-8') as f1:
    temp_result_dic = json.load(f1)
temp_dic_list = {}
for test_file in test_file_list[1:]:
    print("loading ",test_file)
    with open(test_file_path+test_file, 'r', encoding='utf-8') as f:
        dic = json.load(f)
        key_list = dic.keys()
        for key in key_list:
            primary_temp_value = dict(temp_result_dic[key]["Primary_evidence_index"])
            for sent_idx,value in temp_result_dic[key]["Primary_evidence_index"].items():
                if type(primary_temp_value[sent_idx]) == float:
                    primary_temp_value[sent_idx] = [primary_temp_value[sent_idx]]
                primary_temp_value[sent_idx] = primary_temp_value[sent_idx] + [dic[key]["Primary_evidence_index"][sent_idx]]
            temp_result_dic[key]["Primary_evidence_index"] = primary_temp_value

            if len(temp_result_dic[key].keys())==2: # comparision
                secondary_temp_value = dict(temp_result_dic[key]["Secondary_evidence_index"])
                for sent_idx, value in temp_result_dic[key]["Secondary_evidence_index"].items():
                    if type(secondary_temp_value[sent_idx]) == float:
                        secondary_temp_value[sent_idx] = [secondary_temp_value[sent_idx]]
                    secondary_temp_value[sent_idx] = secondary_temp_value[sent_idx] + [dic[key]["Secondary_evidence_index"][sent_idx]]
                temp_result_dic[key]["Secondary_evidence_index"] = secondary_temp_value

result_dic = {}
yuzhi = 0.53
for key in key_list:
    first_pred_idx = []
    primary_temp_value = dict(temp_result_dic[key]["Primary_evidence_index"])
    for pri_idx,arr in primary_temp_value.items():
        ave_score = mean(arr)
        if ave_score > yuzhi:
            first_pred_idx.append(int(pri_idx))
    if len(temp_result_dic[key].keys())==2: # comparision的
        secondary_pred_idx = []
        secondary_temp_value = dict(temp_result_dic[key]["Secondary_evidence_index"])
        for sec_idx, arr in secondary_temp_value.items():
            ave_score = mean(arr)
            if ave_score > yuzhi:
                secondary_pred_idx.append(int(sec_idx))
        result_dic[key] = {"Primary_evidence_index": first_pred_idx, "Secondary_evidence_index": secondary_pred_idx}
    else:
        result_dic[key] = {"Primary_evidence_index": first_pred_idx}

with open(output_file, 'w', encoding='utf-8') as writer:
    json.dump(result_dic, writer, ensure_ascii=False)