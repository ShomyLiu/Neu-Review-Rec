# -*- coding: utf-8 -*-

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from framework import Model
import models


class LitModel(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()

        self.model = Model(opt, getattr(models, opt.model))
        self.opt = opt

        if self.model.net.num_fea != opt.num_fea:
            raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={self.model.net.num_fea}")

    def forward(self, data):
        return self.model(data)

    def compute_loss(self, batch):
        x_data, y_data = self.unpack_input(batch)
        predict_y = self(x_data)
        mse = F.mse_loss(predict_y, y_data)
        mae = F.l1_loss(predict_y, y_data)
        return {'mse': mse, 'mae': mae}

    def training_step(self, batch, batch_idx):
        res = self.compute_loss(batch)
        self.log('train_loss', res['mse'], on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': res['mse']}

    def validation_step(self, batch, batch_idx):
        res = self.compute_loss(batch)
        res = {"val_mse": res['mse'], "val_mae": res['mae']}
        self.log_dict(res, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.opt.use_ddp)
        return res

    def test_step(self, batch, batch_idx):
        res = self.compute_loss(batch)
        metrics = {'test_mse': res['mse'], 'test_mae': res['mae']}
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=self.opt.use_ddp)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        return optimizer

    def unpack_input(self, batch):
        x, y = batch
        uids, iids = list(zip(*x))
        uids = list(uids)
        iids = list(iids)

        user_reviews = self.opt.users_review_list[uids]
        user_item2id = self.opt.user2itemid_list[uids]  # 检索出该user对应的item id
        user_doc = self.opt.user_doc[uids]

        item_reviews = self.opt.items_review_list[iids]
        item_user2id = self.opt.item2userid_list[iids]  # 检索出该item对应的user id
        item_doc = self.opt.item_doc[iids]

        data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
        data = list(map(lambda x: torch.tensor(x, device=self.device).long(), data))
        return data, torch.tensor(y, device=self.device).float()
