# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/nips/SnellSZ17,
  author    = {Jake Snell and
               Kevin Swersky and
               Richard S. Zemel},
  title     = {Prototypical Networks for Few-shot Learning},
  booktitle = {Advances in Neural Information Processing Systems 30: Annual Conference
               on Neural Information Processing Systems 2017, December 4-9, 2017,
               Long Beach, CA, {USA}},
  pages     = {4077--4087},
  year      = {2017},
  url       = {https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html}
}
https://arxiv.org/abs/1703.05175

Adapted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch.
"""
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel
from core.utils.evaluator import ProtoEvaluator


# HPO with shrunken centroids: https://gemfury.com/stream/python:scikit-learn/-/content/neighbors/nearest_centroid.py
class ProtoLayer(nn.Module):
    def __init__(self):
        super(ProtoLayer, self).__init__()
        

    def forward(
        self,
        query_feat,
        support_feat,
        support_target,
        way_num,
        shot_num,
        query_num,
        mode="euclidean",
        experimental=False,
        samples_per_class={}
    ):
        # (batch_size, support_size, feat_dim)
        # (1, 25, 1600)
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        if experimental:
            # (1, 1, 1600)
            # t, wq, c
            query_feat = query_feat.reshape(t, wq, c)
            # (1, 25, 1600)
            # t, w, c
            proto_feat = torch.empty(1, 1, c).to(device='cuda')
            beg_idx = 0
            for class_num, n_samples in samples_per_class.items():
                class_features = support_feat[:, beg_idx:beg_idx+n_samples, :]
                proto_feat = torch.cat([proto_feat, torch.mean(class_features, dim=1)[None, ...]], dim=1)
                beg_idx += n_samples
            
            proto_feat = proto_feat[:, 1:, :]
            
        else:
            # t, wq, c
            query_feat = query_feat.reshape(t, way_num * query_num, c)
            # t, w, c
            # (1, 25, 1600) -> (1, 5, 5, 1600)
            support_feat = support_feat.reshape(t, way_num, shot_num, c)
            # proto_feat = class prototype 
            proto_feat = torch.mean(support_feat, dim=2)

        # return either euclidian distance or cosine similarity
        return {
            # t, wq, 1, c - t, 1, w, c -> t, wq, w
            "euclidean": lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            ),
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.matmul(
                F.normalize(x, p=2, dim=-1),
                torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # FEAT did not normalize the query_feat
            ),
        }[mode](query_feat, proto_feat)

class ProtoNet(MetricModel):
    def __init__(self, **kwargs):
        super(ProtoNet, self).__init__(**kwargs)
        self.proto_layer = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()
        self.evaluator = ProtoEvaluator(self, **kwargs)
        print(f"Test way stats: {self.test_way} Test shot stats: {self.test_shot}")

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output_dict = self.evaluator.eval_batch(episode_size, support_feat, support_target, query_feat, query_target, self.k_fold, self.boot_rounds)

        return output_dict

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        emb = self.emb_func(images)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            emb, mode=1
        )

        output = self.proto_layer(
            query_feat, support_feat, support_target, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss