# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class D_ATTN(nn.Module):
    '''
    Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction
    Rescys 2017
    '''
    def __init__(self, opt):
        super(D_ATTN, self).__init__()
        self.opt = opt
        self.num_fea = 1   # Document

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        u_fea = self.user_net(user_doc)
        i_fea = self.item_net(item_doc)
        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()
        self.opt = opt

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.local_att = LocalAttention(opt.doc_len, win_size=5, emb_size=opt.word_dim, filters_num=opt.filters_num)
        self.global_att = GlobalAttention(opt.doc_len, emb_size=opt.word_dim, filters_num=opt.filters_num)

        fea_dim = opt.filters_num * 4
        self.fc = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(fea_dim, opt.id_emb_size),
        )
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, docs):
        docs = self.word_embs(docs)  # size * 300
        local_fea = self.local_att(docs)
        global_fea = self.global_att(docs)
        r_fea = torch.cat([local_fea]+global_fea, 1)
        r_fea = self.dropout(r_fea)
        r_fea = self.fc(r_fea)

        return torch.stack([r_fea], 1)

    def reset_para(self):
        cnns = [self.local_att.cnn, self.local_att.att_conv[0]]
        for cnn in cnns:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        for cnn in self.global_att.convs:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.fc[-1].weight, -0.1, 0.1)
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)


class LocalAttention(nn.Module):
    def __init__(self, seq_len, win_size, emb_size, filters_num):
        super(LocalAttention, self).__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(win_size, emb_size), padding=((win_size-1)//2, 0)),
            nn.Sigmoid()
        )
        self.cnn = nn.Conv2d(1, filters_num, kernel_size=(1, emb_size))

    def forward(self, x):
        score = self.att_conv(x.unsqueeze(1)).squeeze(1)
        out = x.mul(score)
        out = out.unsqueeze(1)
        out = torch.tanh(self.cnn(out)).squeeze(3)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out


class GlobalAttention(nn.Module):
    def __init__(self, seq_len, emb_size, filters_size=[2, 3, 4], filters_num=100):
        super(GlobalAttention, self).__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(seq_len, emb_size)),
            nn.Sigmoid()
        )
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, emb_size)) for k in filters_size])

    def forward(self, x):
        x = x.unsqueeze(1)
        score = self.att_conv(x)
        x = x.mul(score)
        conv_outs = [torch.tanh(cnn(x).squeeze(3)) for cnn in self.convs]
        conv_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
        return conv_outs
