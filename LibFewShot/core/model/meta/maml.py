# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/icml/FinnAL17,
  author    = {Chelsea Finn and
               Pieter Abbeel and
               Sergey Levine},
  title     = {Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning,
               {ICML} 2017, Sydney, NSW, Australia, 6-11 August 2017},
  series    = {Proceedings of Machine Learning Research},
  volume    = {70},
  pages     = {1126--1135},
  publisher = {{PMLR}},
  year      = {2017},
  url       = {http://proceedings.mlr.press/v70/finn17a.html}
}
https://arxiv.org/abs/1703.03400

Adapted from https://github.com/wyharveychen/CloserLookFewShot.
"""
import torch
from torch import nn

from core.utils import accuracy, check_gpu_memory_usage
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
from core.utils.evaluator import MAMLEvaluator

class MAMLLayer(nn.Module):
    def __init__(self, feat_dim=64, way_num=5) -> None:
        super(MAMLLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


class MAML(MetaModel):
    def __init__(self, inner_param, feat_dim, **kwargs):
        super(MAML, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param
        self.evaluator = MAMLEvaluator(self, **kwargs)

        convert_maml_module(self)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        # output_list = []
        # # iterating through episodes
        # for i in range(episode_size):
        #     episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
        #     episode_support_target = support_target[i].reshape(-1)
        #     episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
        #     # episode_query_target = query_target[i].reshape(-1)

        #     # fine-tuning model with support set
        #     self.set_forward_adaptation(episode_support_image, episode_support_target)

        #     # testing model on the query set
        #     output = self.forward_output(episode_query_image)

        #     output_list.append(output)

        # output = torch.cat(output_list, dim=0)
        # acc = accuracy(output, query_target.contiguous().view(-1))

        # output_dict = self.evaluator.eval_batch(episode_size, support_image, support_target, query_image, query_target, self.k_fold)
        output_dict = self.evaluator.eval_batch(episode_size, support_image, support_target, query_image, query_target, self.k_fold, self.boot_rounds)
        return output_dict
        # return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_targets = query_targets[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, support_target):
        lr = self.inner_param["lr"]
        for parameter in self.parameters():
            parameter.fast = None
        fast_parameters = list(self.parameters())

        self.emb_func.train()
        self.classifier.train()
        # for i in range(
        #     self.inner_param["train_iter"]
        #     if self.training
        #     else self.inner_param["test_iter"]
        # ):
        for i in range(
            self.inner_param["train_iter"]
            if self.training
            else self.inner_param["test_iter"]
        ):
            # print(f"Iteration number {i}")
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            # if not  self.training:
            #     print(f"Iter {i+1} - loss: {loss}")
            # if self.training:
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=False, retain_graph=False)
                # grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - lr * grad[k]
                fast_parameters.append(weight.fast)
            
            # loss.backward()
            # for k, weight in enumerate(self.parameters()):
                # grad[k].zero_()
            # fast_parameters.backward()
        
        # del grad
        # del fast_parameters
        # for k, weight in enumerate(self.parameters()):

        return loss

    def set_forward_cross_validation(self, support_feat, support_target, query_feat, query_target, k):
        # iterating through the k possible splits
        for idx in range(k):
            cv_support_feat = support_feat.clone()
            cv_support_target = support_target.clone()
            cv_support_feat_test = cv_support_feat[idx]
            cv_support_target_test = cv_support_target[idx]
            cv_support_feat = torch.cat([cv_support_feat[0:idx], cv_support_feat[idx+1:]])
            cv_support_target = torch.cat([cv_support_target[0:idx], cv_support_target[idx+1:]])

            self.set_forward_adaptation(cv_support_feat, cv_support_target)

            # testing model on the query set
            output = self.forward_output(query_feat)
        return 