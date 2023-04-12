# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils_biolinkbert_entailment import get_time_dif
from pytorch_pretrained.optimization import BertAdam

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0
    dev_best_f1 = float('-inf')
    dev_sec_f1 = float('-inf')
    dev_th_f1 = float('-inf')
    last_improve = 0
    flag = False
    model.train()
    with open(config.output_filename, 'a') as writer:
        for epoch in range(config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs), file=writer)
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = 0
                for i in range(len(labels)):
                    try:
                        output_temp = outputs[i, :]
                    except:
                        print(outputs)
                        input()
                    output_temp = output_temp.reshape(1, -1)
                    label_temp = labels[i:i+1]
                    loss_temp = F.cross_entropy(output_temp, label_temp)
                    loss += loss_temp
                loss = loss / len(labels)
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc, dev_loss, f1 = evaluate(config, model, dev_iter)
                    if f1 > dev_best_f1:
                        dev_best_f1 = f1
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    elif dev_sec_f1 < f1 < dev_best_f1:
                        dev_sec_f1 = f1
                        torch.save(model.state_dict(), config.save_path2)
                        improve = '**'
                        last_improve = total_batch
                    elif dev_th_f1 < f1 < dev_sec_f1:
                        dev_th_f1 = f1
                        torch.save(model.state_dict(), config.save_path3)
                        improve = '***'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  f1 score: {5:>6.2%}, Time: {6} {7}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, f1, time_dif, improve))
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, f1, time_dif, improve), file = writer)
                    model.train()
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion,f1_score = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print("f1:", f1_score)


def label2index(label_list):
    label_index = []
    for i in range(len(label_list)):
        if label_list[i] == 1:
            label_index.append(i)
    return label_index

def indicators(pred_index,ground_index):
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(ground_index)):
        if ground_index[i] in pred_index:
            TP += 1
        if ground_index[i] not in pred_index:
            FN += 1
    for j in range(len(pred_index)):
        if pred_index[j] not in ground_index:
            FP += 1
    if TP+FP == 0:
        TP = 0.0001
    
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1_score = 2*(p * r)/(p + r)

    return p,r,f1_score

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = 0
            for i in range(len(labels)):
                output_temp = outputs[i, :]
                output_temp = output_temp.reshape(1, -1)
                label_temp = labels[i:i + 1]
                loss_temp = F.cross_entropy(output_temp, label_temp)
                loss += loss_temp
            loss = loss / len(labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    pred_index = label2index(predict_all)
    ground_index = label2index(labels_all)

    if not pred_index:
        pred_index = [0]

    p,r,f1_score = indicators(pred_index,ground_index)

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion, f1_score
    return acc, loss_total / len(data_iter), f1_score
