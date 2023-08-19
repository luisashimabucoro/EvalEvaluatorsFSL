# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/ChenLKWH19,
  author    = {Wei{-}Yu Chen and
               Yen{-}Cheng Liu and
               Zsolt Kira and
               Yu{-}Chiang Frank Wang and
               Jia{-}Bin Huang},
  title     = {A Closer Look at Few-shot Classification},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HkxLXnAcFQ}
}
https://arxiv.org/abs/1904.04232

Adapted from https://github.com/wyharveychen/CloserLookFewShot.
"""
import pdb
import torch
from torch import nn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.linear_model import LogisticRegression

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from core.utils.evaluator import BaselineEvaluator


class Baseline(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(Baseline, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param

        # self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()

        self.evaluator = BaselineEvaluator(self, **kwargs)

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)
        output_dict = self.evaluator.eval_batch(episode_size, support_feat, support_target, query_feat, query_target, self.k_fold, self.boot_rounds)

        return output_dict


    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = LogisticRegression(max_iter=self.inner_param["inner_train_iter"])

        classifier.fit(support_feat.cpu(), support_target.cpu())

        query_output = classifier.predict_proba(query_feat.cpu())
        
        return classifier, query_output

    def set_forward_adaptation_og(self, support_feat, support_target, query_feat):
        classifier = nn.Linear(self.feat_dim, self.way_num)
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)

        # number of epochs to train new classifier
        for epoch in range(self.inner_param["inner_train_iter"]):
        # for epoch in range(300):
            rand_id = torch.randperm(support_size)
            # split support set into batches
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                select_id = rand_id[
                    i : min(i + self.inner_param["inner_batch_size"], support_size)
                ]
                batch = support_feat[select_id]
                target = support_target[select_id]

                output = classifier(batch)

                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        query_output = classifier(query_feat)
        
        return classifier, query_output
