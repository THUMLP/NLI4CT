import json
import math
def find_best_thres(pred_pair):
    TP, FP, TN = 0,0,0
    for one in pred_pair:
        if one[1] == 1: # true
            TN += 1
    sorted_pred_pair = sorted(pred_pair, key=lambda x: x[0], reverse=True)
    best_thres = 1
    best_f1 = 0
    for one in sorted_pred_pair:
        pred, label = one[0], one[1]
        if one[1] == 1:
            TP += 1
            TN -= 1
        else:
            FP += 1
        rec = TP/(TP+TN+1e-8)
        pre = TP/(TP+FP+1e-8)
        f1 = 2*rec*pre/(rec+pre+1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thres = pred
    return best_thres, best_f1
def softmax(input, temp):
    exp_input = [math.exp(one/temp) for one in input]
    sumed_input = 0
    for i in exp_input:
        sumed_input += i
    output = [one/sumed_input for one in exp_input]
    return output

test_name_list = [
        'std_test_result_1024_tf_bi_mul',
        'std_test_result_scifive',
        'std_test_result_512_bi_bi_mul',
        'std_test_result_512_tf_bi_cl',
        ]


gold_data = json.load(open('test.json','r'))

out_file = {}
thres = 0.57
pro, con = 0,0
for name in test_name_list:
    for i in range(10):
        test_temp_name = '{}_{}'.format(name, i)
        data = json.load(open(test_temp_name+'.json','r'))
        if len(data[0]) > 5:
            data = [one[:-1] for one in data]
        data = {one[0]:float(one[-1]) for one in data}
        for guid in data:
            if guid not in out_file:
                out_file[guid] = {"Prediction":0}
            section = gold_data[guid]['Section_id']
            assert data[guid] <= 1
            
            out_file[guid]["Prediction"] += 1/10 * 1/len(test_name_list) * data[guid]

# print(out_file)
# load joint inference module's results
pair_result = json.load(open('std_test_result_biolinkbert_joint_inference.json','r'))
pair_detect = {}
for one in pair_result:
    guid0 = one[1]
    guid1 = one[2]
    key = guid0 + ' ' + guid1
    key2 = guid1 + ' ' + guid0
    score = float(one[-1])
    if key in pair_detect:
        pair_detect[key] += score/2
    elif key2 in pair_detect:
        pair_detect[key2] += score/2
    else:
        pair_detect[key] = 0
        pair_detect[key] += score/2
for key in pair_detect:
    score = pair_detect[key]
    tmp = key.split(' ')
    guid0, guid1 = tmp[0],tmp[1]
    if score < 0.5:
        if ((out_file[guid0]["Prediction"]>thres and out_file[guid1]["Prediction"]>thres) or (out_file[guid0]["Prediction"]<=thres and out_file[guid1]["Prediction"]<=thres)): # contradiction detected
            if out_file[guid0]["Prediction"] > out_file[guid1]["Prediction"]:
                out_file[guid0]["Prediction"] = 'Entailment'
                out_file[guid1]["Prediction"] = 'Contradiction'
            else:
                out_file[guid0]["Prediction"] = 'Contradiction'
                out_file[guid1]["Prediction"] = 'Entailment'
        else:
            print("{} {}".format(out_file[guid0]["Prediction"], out_file[guid1]["Prediction"]))


for guid in out_file:
    # assert out_file[guid]["Prediction"] < 1
    if out_file[guid]["Prediction"] not in ['Entailment', 'Contradiction']:
        # print(out_file[guid]["Prediction"])
        out_file[guid]["Prediction"] = 'Entailment' if out_file[guid]["Prediction"] > thres else 'Contradiction'

json.dump(out_file, open("test_ensemble_avg.json",'w'),indent=4)