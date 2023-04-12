# coding: UTF-8
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# jinziyu
class Config(object):

    """配置参数"""
    def __init__(self, dataset):

        self.model_name = 'biolinkbert'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.save_path2 = dataset + '/saved_dict_2/' + self.model_name + '.ckpt'
        self.save_path3 = dataset + '/saved_dict_3/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_filename = dataset + "log.txt"  # todo

        self.require_improvement = 40000
        self.num_classes = len(self.class_list)
        self.num_epochs = 50  # epoch
        self.batch_size = 1  # batch
        self.pad_size = 512
        self.learning_rate = 3e-6  # lr
        self.bert_path = './biolinkbert'
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 1024
        self.dropout = 0
        self.rnn_hidden = 1024
        self.num_layers = 1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.device = config.device
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.hidden_size)

    def forward(self, x):
        # print(x)
        # print(len(x))
        # input()
        context = x[0]  # 输入的句子 batch, seq_len 1,512
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        # token_type_ids = x[3] # 做cls需要注释
        # print("context: ", x[0])
        # print("x[0].shape: ", x[0].shape)
        # print("x[2].shape: ", x[2].shape)
        # print("x[3].shape: ", x[3].shape)
        # input()
        # out = self.bert(context, attention_mask=mask,token_type_ids=token_type_ids)
        out = self.bert(context, attention_mask=mask)
        # print(out)
        # input()
        out = out.last_hidden_state # (1,512,1024)

        # example_span_statement = [1,30]
        # example_span_evi = [[31,100],[101,512]]
        all_sent = x[4].cpu().numpy()
        span_statement = all_sent[0][0]
        span_evi = all_sent[0][1:]
        # sent_evi_idx = int(x[5][0])
        # print("all_sent: ",all_sent)
        # print("sent_evi_idx: ",sent_evi_idx)
        # print("span_statement: ",span_statement)
        # print("span_evi: ",span_evi)
        # input()

        # 句子级别表征融合
        sentlevel_emb = torch.tensor([]).to(self.device)  # todo
        statement_emb_start = span_statement[0]
        statement_emb_end = span_statement[1] + 1
        statement_emb = out[:, statement_emb_start:statement_emb_end, :]  # torch.Size([1, 39, 1024])
        # print(statement_emb)
        # print(statement_emb.shape)
        # input()

        # max pooling方法
        statement_emb_max = torch.max(statement_emb, dim=1).values
        statement_emb_max = statement_emb_max.reshape(1, 1, -1)
        sentlevel_emb = torch.cat((sentlevel_emb, statement_emb_max))  # torch.Size([1, 1, 1024])

        for i in range(len(span_evi)):
            temp_span_start = span_evi[i][0]
            temp_span_end = span_evi[i][1] + 1
            temp_out = out[:, temp_span_start:temp_span_end, :]

            # max pooling
            temp_out_max = torch.max(temp_out, dim=1).values
            temp_out_max = temp_out_max.reshape(1, 1, -1)
            sentlevel_emb = torch.cat((sentlevel_emb, temp_out_max), 1)  # todo

        # print(sentlevel_emb.shape)
        sentlevel_emb, _ = self.lstm(sentlevel_emb)
        sentlevel_emb = self.fc_rnn(sentlevel_emb[:, :, :])
        # print(sentlevel_emb.shape)
        # input()

        # print(sentlevel_emb.shape)
        # X = self.attention(sentlevel_emb)  # 所有句子的embedding, 要求维度是num_sent,batch,hidden_dim)  111
        # sentlevel_emb_res = sentlevel_emb + X   111
        # print(sentlevel_emb_res.shape)
        # input()


        # # token级别表征融合 尝试1
        all_state_evi_emb = torch.tensor([]).to(self.device)
        for j in range(len(span_evi)):
            evidence_sent_emb_start, evidence_sent_emb_end = span_evi[j]
            evidence_sent_emb_end += 1
            evidence_sent_emb = out[:,evidence_sent_emb_start:evidence_sent_emb_end,:]
            # print(statement_emb.shape)
            # print(evidence_sent_emb.shape)
            # input()
            statement_evi_emb = torch.cat((statement_emb, evidence_sent_emb),1) # statement 1,50,1024
            # print(statement_evi_emb.shape)
            # input()

            statement_evi_emb_res = statement_evi_emb
            statement_evi_emb_max = torch.max(statement_evi_emb_res, dim=1).values
            statement_evi_emb_max = statement_evi_emb_max.reshape(1, 1, -1)
            all_state_evi_emb = torch.cat((all_state_evi_emb, statement_evi_emb_max), 1)

        # print("all_state_evi_emb", all_state_evi_emb.shape)  # torch.Size([1, 6, 1024])
        # input()


        # 两个相加
        out_list = torch.tensor([]).to(self.device)
        for k in range(len(span_evi)):
            # print(len(span_evi))
            # print(sentlevel_emb[:,k+1,:].shape)
            # print(all_state_evi_emb[:,k,:].shape)
            sentlevel_emb_temp = torch.tensor([]).to(self.device)
            sentlevel_emb_temp = torch.cat((sentlevel_emb_temp, sentlevel_emb[:,k+1,:]))
            # print("sentlevel_emb_temp.shape", sentlevel_emb_temp.shape)
            # input()
            sentlevel_emb_temp = torch.cat((sentlevel_emb_temp, all_state_evi_emb[:,k,:]), 1)
            # print("sentlevel_emb_temp.shape",sentlevel_emb_temp.shape)
            out = self.fc(sentlevel_emb_temp)
            # print(out)
            # print(out.shape) # torch.Size([1, 2])
            # input()
            out_list = torch.cat((out_list, out), 0)

        # print(out_list)
        # print(out_list.shape)
        # input()
        return out_list
