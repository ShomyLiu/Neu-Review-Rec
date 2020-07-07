# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NARRE(nn.Module):
    '''
    NARRE: WWW 2018
    '''
    def __init__(self, opt):
        super(NARRE, self).__init__()
        self.opt = opt
        self.num_fea = 2  # ID + Review

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        u_fea = self.user_net(user_reviews, uids, user_item2id)
        i_fea = self.item_net(item_reviews, iids, item_user2id)
        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)

        self.review_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size, bias=False)
        self.attention_linear = nn.Linear(self.opt.id_emb_size, 1)
        self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)

        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, reviews, ids, ids_list):
        # --------------- word embedding ----------------------------------
        reviews = self.word_embs(reviews)  # size * 300
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)

        id_emb = self.id_embedding(ids)

        u_i_id_emb = self.u_i_id_embedding(ids_list)

        # --------cnn for review--------------------
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)
        fea = fea.view(-1, r_num, fea.size(1))

        # ------------------linear attention-------------------------------
        rs_mix = F.relu(self.review_linear(fea) + self.id_linear(F.relu(u_i_id_emb)))
        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, 1)
        r_fea = fea * att_weight
        r_fea = r_fea.sum(1)
        r_fea = self.dropout(r_fea)

        return torch.stack([id_emb, self.fc_layer(r_fea)], 1)

    def reset_para(self):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-0.1, b=0.1)
