
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertEncoder, BertPooler
from typing import Optional
from packaging import version
from torch_scatter import scatter_max


class BertEmbeddings_1024(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.new_position_embeddings = nn.Embedding(1024, config.hidden_size)
        self.register_buffer("new_position_ids", torch.arange(1024).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.new_position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def update_position_embedding(self,init_type):
        position_emb_weight = self.position_embeddings.weight
        # new_position_embeddings = nn.Embedding(1024, position_emb_weight.size(1))
        self.new_position_embeddings.weight[:512,:].data.copy_(position_emb_weight)
        if init_type == 0:
            self.new_position_embeddings.weight[512:,:].data.copy_(position_emb_weight)
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.new_position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.new_position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.new_position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModel_1024(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = BertEmbeddings_1024(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
    def update_position_embedding(self, init_type=0):
        self.embeddings.update_position_embedding(init_type)


class model_tf_bi(nn.Module):
    '''
    尝试新的sentence-level interaction: 先拼接后交互
    '''
    def __init__(self, bert_model, config_path='biolinkbert-large-mnli-snli', num_layers=1, pool='max', num_head=16):
        super(model_tf_bi, self).__init__()
        self.bert_hidden_dim = bert_model.config.hidden_size
        self.dropout = nn.Dropout(0.5)
        self.num_labels = 2
        self.pred_model = bert_model
        # self.sent_inter = nn.LSTM(input_size=self.bert_hidden_dim, 
        #                         hidden_size=self.bert_hidden_dim, 
        #                         num_layers=1, 
        #                         batch_first=True, 
        #                         bidirectional=True)
        self.config = BertConfig.from_json_file(config_path)
        self.config.output_attentions = True
        self.config.num_hidden_layers = num_layers
        self.config.num_attention_heads = num_head
        self.sent_inter = BertModel(self.config)
        self.token_inter = nn.LSTM(input_size=self.bert_hidden_dim, 
                                hidden_size=self.bert_hidden_dim, 
                                num_layers=1, 
                                batch_first=True, 
                                bidirectional=True)
        self.fc_post_lstm = nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim)
        # self.fc_post_sent_inter = nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim)
        # self.st_evi_merge = nn.Linear(self.bert_hidden_dim * 2, self.bert_hidden_dim)
        self.task_b_cls = nn.Linear(self.bert_hidden_dim * 2, 1)
        # self.single_neg = nn.Sequential(nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim),nn.GELU(), nn.Linear(self.bert_hidden_dim, 2))
        self.multi_cls = nn.Sequential(nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),nn.GELU(), nn.Linear(self.bert_hidden_dim, 2))

        self.pool = pool
        if pool == 'bilstm':
            self.token_pooler = nn.LSTM(input_size=self.bert_hidden_dim, 
                                hidden_size=self.bert_hidden_dim, 
                                num_layers=1, 
                                batch_first=True, 
                                bidirectional=True)
            self.token_pooler_fc = nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim)
        elif pool == 'transformer':
            self.token_pooler = BertModel(self.config)
        # self.mode_switch = nn.Linear(self.bert_hidden_dim, 2)

    def forward(self, inputs):
        input_ids, att_mask, segment_ids = inputs
        input_ids = input_ids.view(1, -1)
        att_mask = att_mask.view(1, -1)
        segment_ids = segment_ids.view(1, -1)
        outputs = self.pred_model(input_ids, att_mask) # encoding
        inputs_hiddens = outputs.last_hidden_state  # 1, max_len, emb_size
        # inputs = outputs.pooler_output
        # mode_logits = self.mode_switch(inputs)
        # mode_prob = torch.softmax(mode_logits.view(-1,1),dim=0)
        evi_num = int(torch.max(segment_ids, dim=1).values)+1
        sentence_repre = []
        for i in range(evi_num + 1):
            sent_inds = torch.where(segment_ids==i-1)
            sent_embed = inputs_hiddens[sent_inds[0], sent_inds[1], :]
            sentence_repre.append(sent_embed)
        # sent_level: max_pooling
        if self.pool == 'max':
            sentence_repre_pooled = [torch.max(one, dim=0).values for one in sentence_repre]
        elif self.pool =='transformer':
            sentence_repre_pooled = []
            for one in sentence_repre:
                tmp = self.token_pooler(inputs_embeds=one.unsqueeze(0)).pooler_output
                sentence_repre_pooled.append(tmp.squeeze(0))
        elif self.pool == 'bilstm':
            sentence_repre_pooled = []
            for one in sentence_repre:
                _, (h_n, c_n) = self.token_pooler(one.unsqueeze(0))
                h_n = h_n.view(1,-1)
                sentence_repre_pooled.append(self.token_pooler_fc(h_n).squeeze(0))
        else:
            sentence_repre_pooled = [torch.mean(one, dim=0) for one in sentence_repre]
        sentence_repre_pooled = torch.stack(sentence_repre_pooled,dim=0)  # evi_num+1, emb_size
        # sent_level interact: bilstm
        # sentence_repre_pooled = sentence_repre_pooled.unsqueeze(0)
        # evi_pooled = sentence_repre_pooled[1:,:]
        # statement_pooled_repeat = sentence_repre_pooled[0,:].view(1,-1).repeat(evi_num,1)
        sent_context_result = self.sent_inter(inputs_embeds=sentence_repre_pooled.unsqueeze(0))
        sent_context_post = sent_context_result.last_hidden_state
        sent_context_attentions = sent_context_result.attentions[0]
        attention_out = sent_context_attentions.squeeze(0).mean(0)
        evi_sent_level = sent_context_post.squeeze(0)[1:,:] # evi_num, emb_size
        nli_repre = sent_context_post.squeeze(0)[0,:].view(1,-1) # sent_level
        multi_pred = torch.softmax(self.multi_cls(nli_repre), dim=-1) # 1, 2
        
        evi_token_level = []
        single_pred = []
        # token_level interact: bilstm
        for i in range(evi_num):
            tmp_token_emb = torch.cat([sentence_repre[0],sentence_repre[i+1]],dim=0)
            tmp_token_emb = tmp_token_emb.unsqueeze(0) # 1, sent_len+evi_len, emb_size
            _, (h_n, c_n) = self.token_inter(tmp_token_emb)
            h_n = h_n.reshape(1, -1)
            # pred = self.single_neg(h_n) # contradiction or not sure?
            # single_pred.append(torch.softmax(pred,dim=-1))
            h_n_post = self.fc_post_lstm(h_n)
            evi_token_level.append(h_n_post)
        evi_token_level = torch.cat(evi_token_level, dim=0)
        # single_pred = torch.cat(single_pred, dim=0) # evi_num, 2
        # best_pred_ind = torch.max(single_pred[:,0], dim=0).indices
        # pessi_pred = single_pred[best_pred_ind,:]   # if one reject, then reject
        # final_pred = torch.cat([multi_pred, pessi_pred.view(1,-1)],dim=0)
        # mixed_pred = torch.sum(final_pred*mode_prob, dim=0, keepdim=True)
        mixed_pred_log = torch.log(multi_pred)
        evi_logits = self.task_b_cls(torch.cat([evi_sent_level, evi_token_level],dim=-1)).view(-1)
        return mixed_pred_log, evi_logits, attention_out

class model_tf_bi_batch(model_tf_bi):
    '''
    尝试新的sentence-level interaction: 先拼接后交互
    '''
    def __init__(self, bert_model, config_path='biolinkbert-large-mnli-snli', num_layers=1, pool='max', num_head=16):
        super(model_tf_bi_batch, self).__init__(bert_model, config_path, num_layers, pool, num_head)
        self.task_b_cls = nn.Linear(self.bert_hidden_dim, 1)

    def forward(self, inputs):
        input_ids, att_mask, segment_ids = inputs
        outputs = self.pred_model(input_ids, att_mask) # encoding
        inputs_hiddens = outputs.last_hidden_state  # batch_size, max_len, emb_size
        # inputs = outputs.pooler_output
        # mode_logits = self.mode_switch(inputs)
        # mode_prob = torch.softmax(mode_logits.view(-1,1),dim=0)
        evi_num = torch.max(segment_ids, dim=1).values-1
        
        sentence_repre_pooled, _ = scatter_max(inputs_hiddens, segment_ids, dim=1) # batch, max_evi_num, emb_size
        sentence_repre_pooled = sentence_repre_pooled[:, 1:, :]
        sent_level_attention = []
        for i in evi_num:
            use_num =int(i+1)
            sent_level_attention.append([1]*use_num+[0]*(sentence_repre_pooled.size(1)-use_num))
        sent_level_attention = torch.tensor(sent_level_attention, dtype=torch.long, device=input_ids.device)
        # sent_level interact: bilstm
        # sentence_repre_pooled = sentence_repre_pooled.unsqueeze(0)
        # evi_pooled = sentence_repre_pooled[1:,:]
        # statement_pooled_repeat = sentence_repre_pooled[0,:].view(1,-1).repeat(evi_num,1)
        sent_context_result = self.sent_inter(inputs_embeds=sentence_repre_pooled, attention_mask=sent_level_attention)
        sent_context_post = sent_context_result.last_hidden_state
        evi_sent_level = sent_context_post[:, 1:,:] # batch, evi_num, emb_size
        nli_repre = sent_context_post[:,0,:] # batch, sent_level
        multi_pred = torch.softmax(self.multi_cls(nli_repre), dim=-1) # batch, 2
        
        # single_pred = torch.cat(single_pred, dim=0) # evi_num, 2
        # best_pred_ind = torch.max(single_pred[:,0], dim=0).indices
        # pessi_pred = single_pred[best_pred_ind,:]   # if one reject, then reject
        # final_pred = torch.cat([multi_pred, pessi_pred.view(1,-1)],dim=0)
        # mixed_pred = torch.sum(final_pred*mode_prob, dim=0, keepdim=True)
        mixed_pred_log = torch.log(multi_pred)
        evi_logits = self.task_b_cls(evi_sent_level).squeeze(-1)
        return mixed_pred_log, evi_logits, nli_repre

class model_bi_bi(nn.Module):
    def __init__(self, bert_model):
        super(model_bi_bi, self).__init__()
        self.bert_hidden_dim = bert_model.config.hidden_size
        self.dropout = nn.Dropout(0.5)
        self.num_labels = 2
        self.pred_model = bert_model
        self.sent_inter = nn.LSTM(input_size=self.bert_hidden_dim, 
                                hidden_size=self.bert_hidden_dim, 
                                num_layers=1, 
                                batch_first=True, 
                                bidirectional=True)
        self.token_inter = nn.LSTM(input_size=self.bert_hidden_dim, 
                                hidden_size=self.bert_hidden_dim, 
                                num_layers=1, 
                                batch_first=True, 
                                bidirectional=True)
        self.fc_post_lstm = nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim)
        self.fc_post_sent_inter = nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim)
        self.task_b_cls = nn.Linear(self.bert_hidden_dim * 2, 1)
        # self.single_neg = nn.Sequential(nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim),nn.GELU(), nn.Linear(self.bert_hidden_dim, 2))
        self.multi_cls = nn.Sequential(nn.Linear(2*self.bert_hidden_dim, self.bert_hidden_dim),nn.GELU(), nn.Linear(self.bert_hidden_dim, 2))
        # self.mode_switch = nn.Linear(self.bert_hidden_dim, 2)

    def forward(self, inputs):
        input_ids, att_mask, segment_ids = inputs
        input_ids = input_ids.view(1, -1)
        att_mask = att_mask.view(1, -1)
        segment_ids = segment_ids.view(1, -1)
        outputs = self.pred_model(input_ids, att_mask) # encoding
        inputs_hiddens = outputs.last_hidden_state  # 1, max_len, emb_size
        inputs = outputs.pooler_output
        # mode_logits = self.mode_switch(inputs)
        # mode_prob = torch.softmax(mode_logits.view(-1,1),dim=0)
        evi_num = int(torch.max(segment_ids, dim=1).values)+1
        sentence_repre = []
        for i in range(evi_num + 1):
            sent_inds = torch.where(segment_ids==i-1)
            sent_embed = inputs_hiddens[sent_inds[0], sent_inds[1], :]
            sentence_repre.append(sent_embed)
        # sent_level: max_pooling
        sentence_repre_pooled = [torch.max(one, dim=0).values for one in sentence_repre]
        sentence_repre_pooled = torch.stack(sentence_repre_pooled,dim=0)  # evi_num+1, emb_size
        # sent_level interact: bilstm
        sentence_repre_pooled = sentence_repre_pooled.unsqueeze(0)
        sent_context, (h_n,c_n) = self.sent_inter(sentence_repre_pooled)
        sent_context_post = self.fc_post_sent_inter(sent_context)
        evi_sent_level = sent_context_post.squeeze(0)[1:,:] # evi_num, emb_size
        h_n = h_n.reshape(1, -1) # sent_level
        multi_pred = torch.softmax(self.multi_cls(h_n), dim=-1) # 1, 2
        
        evi_token_level = []
        single_pred = []
        # token_level interact: bilstm
        for i in range(evi_num):
            tmp_token_emb = torch.cat([sentence_repre[0],sentence_repre[i+1]],dim=0)
            tmp_token_emb = tmp_token_emb.unsqueeze(0) # 1, sent_len+evi_len, emb_size
            _, (h_n, c_n) = self.token_inter(tmp_token_emb)
            h_n = h_n.reshape(1, -1)
            # pred = self.single_neg(h_n) # contradiction or not sure?
            # single_pred.append(torch.softmax(pred,dim=-1))
            h_n_post = self.fc_post_lstm(h_n)
            evi_token_level.append(h_n_post)
        evi_token_level = torch.cat(evi_token_level, dim=0)
        # single_pred = torch.cat(single_pred, dim=0) # evi_num, 2
        # best_pred_ind = torch.max(single_pred[:,0], dim=0).indices
        # pessi_pred = single_pred[best_pred_ind,:]   # if one reject, then reject
        # final_pred = torch.cat([multi_pred, pessi_pred.view(1,-1)],dim=0)
        # mixed_pred = torch.sum(final_pred*mode_prob, dim=0, keepdim=True)
        mixed_pred_log = torch.log(multi_pred)
        evi_logits = self.task_b_cls(torch.cat([evi_sent_level, evi_token_level],dim=-1)).view(-1)
        return mixed_pred_log, evi_logits