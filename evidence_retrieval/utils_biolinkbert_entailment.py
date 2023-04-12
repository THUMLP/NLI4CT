# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta


def build_dataset(config):
    def dynamic_dataprocess(evi_sent_list, pad_size, statement):
        num_evi_sent = len(evi_sent_list)
        evi_sent_dict = {}
        for i in range(len(evi_sent_list)):
            evi_sent_dict[i] = config.tokenizer.tokenize(evi_sent_list[i])   # index：token list
        
        num_cls = 1
        num_sep = num_evi_sent + 1
        len_statement = len(config.tokenizer.tokenize(statement))
        aver_len = int((pad_size - num_cls - num_sep - len_statement)/num_evi_sent)  
        final_evi_sent_list = [0] * num_evi_sent
        last_num_evi_sent = num_evi_sent
        evi_sent_order=sorted(evi_sent_dict.items(),key=lambda x:len(x[1]),reverse=False)

        for i in range(num_evi_sent):
            max_sent = evi_sent_order[-1][1]
            max_sent_index = evi_sent_order[-1][0]
            min_sent = evi_sent_order[0][1]
            min_sent_index = evi_sent_order[0][0]
            if len(max_sent) <= aver_len: 
                for one in evi_sent_order:
                    final_evi_sent_list[one[0]] = one[1]
                return final_evi_sent_list
                
            if len(min_sent) >= aver_len:
                for one in evi_sent_order:
                    final_evi_sent_list[one[0]] = one[1][0:aver_len]
                return final_evi_sent_list

            if len(min_sent) < aver_len and len(max_sent) > aver_len:
                final_evi_sent_list[min_sent_index] = min_sent
                evi_sent_order.pop(0)
                last_num_evi_sent -= 1
                aver_len = int((aver_len * last_num_evi_sent + aver_len - len(min_sent)) / last_num_evi_sent)


    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                input_list = lin.split('\t')
                label_tmp = input_list[-1]
                label = []
                for l in label_tmp:
                    if l == "0" or l == "1":
                        label.append(int(l))
                statement = input_list[-2]
                evi_sent_list = input_list[:-2]
                num_evi_sent = len(evi_sent_list)
                token_statement = config.tokenizer.tokenize(statement)
                evi_sent_span = []
                section = " ".join(evi_sent_list)
                # cls+state+sep+sen1+sep+sen2+...+sep+senn形式
                len_statement = len(statement)
                num_sep = num_evi_sent + 1
                num_cls = 1
                final_evi_sent_list = dynamic_dataprocess(evi_sent_list, pad_size, statement)

                # cls+state+sep+sen1+sep+sen2+...+sep+senn转token
                per_data_token = []
                per_data_token.append('[CLS]')
                for per_token in token_statement:
                    per_data_token.append(per_token)
                if final_evi_sent_list != None and len(final_evi_sent_list) > 1:
                    for sen in final_evi_sent_list:
                        per_data_token.append('[SEP]')
                        for one_token in sen:
                            per_data_token.append(one_token)
                per_data_token.append('[SEP]')

                # token2id
                per_data_token_ids = config.tokenizer.convert_tokens_to_ids(per_data_token)

                cls_sep_index_list = []
                cls_sep_index_list.append(0)
                cls_sep_index_list.extend([i for i,x in enumerate(per_data_token_ids) if x == 3])


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
                contents.append((per_data_token_ids, label, seq_len, mask, segment_ids, state_sen_interval_list))  # todo
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = y[0]

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        seg = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        interval = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        return (x, seq_len, mask, seg, interval), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
