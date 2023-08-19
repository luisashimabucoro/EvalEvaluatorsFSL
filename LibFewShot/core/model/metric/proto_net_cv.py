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

from core.utils import accuracy_cv
from .metric_model import MetricModel
from core.utils.evaluator import ProtoCVEvaluator
import numpy as np
from sklearn.neighbors import NearestCentroid
from core.utils import accuracy, get_shuffle, check_gpu_memory_usage, create_split_list


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
        experimental=True,
        samples_per_class={}
    ):
        # (batch_size, support_size, feat_dim)
        # (1, 25, 1600)
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        #?===================================================================== 
        best_acc = 0
        best_shrinkage = 0
        if experimental:
            episode_shuffle_idx = get_shuffle(way_num, shot_num)
            split_size_list = create_split_list(way_num * shot_num, 5)
            episode_shuffle_idx = list(torch.split(episode_shuffle_idx, split_size_list))
            shrink_scale_list = np.linspace(0,1,11)

            for shrinkage in shrink_scale_list:
                cv_support_feat = support_feat[0].clone()
                cv_support_target = support_target[0].clone()
                cv_acc = []
                
                for idx in range(5):
                    shuffle_idx = episode_shuffle_idx.copy()
                    test_idx = shuffle_idx.pop(idx)
                    train_idx, _ = torch.sort(torch.cat(shuffle_idx))
                    cv_support_feat = support_feat[0].clone()
                    cv_support_target = support_target[0].clone()
                    cv_support_feat_test = cv_support_feat[test_idx]
                    cv_support_target_test = cv_support_target[test_idx]
                    cv_support_feat = cv_support_feat[train_idx]
                    cv_support_target = cv_support_target[train_idx]

                    # defining number of samples per class used to determine the class prototypes
                    train_samples_per_class = {class_num : shot_num for class_num in range(way_num)}
                    for idx in cv_support_target_test:
                        train_samples_per_class[int(idx)] -= 1
                    
                    cv_support_feat_np = cv_support_feat.cpu().detach().numpy()
                    if (np.isfinite(cv_support_feat_np) == False).any():
                        print("Non-finite value is present.")
                        print(cv_support_feat_np)
                    cv_support_feat_np[~np.isfinite(cv_support_feat_np)] = 0.1
                    cv_support_target_np = cv_support_target.cpu().detach().numpy()
                    cv_support_target_np[~np.isfinite(cv_support_target_np)] = 0
                    # cv_support_feat_test_np = cv_support_feat_test.cpu().detach().numpy()
                    # cv_support_feat_test_np[~np.isfinite(cv_support_feat_test_np)] = 0
                    # cv_support_target_test_np = cv_support_target_test.cpu().detach().numpy()
                    # cv_support_target_test_np[~np.isfinite(cv_support_target_test_np)] = 0
                    # if (~np.isfinite(cv_support_feat_test)).any():
                        # print("Existe non-finite value.")

                    # cv_support_feat_np[np.isnan(cv_support_feat_np)] = np.median(cv_support_feat_np[~np.isnan(cv_support_feat_np)])
                    # cv_support_feat_test_np[np.isnan(cv_support_feat_test_np)] = np.median(cv_support_feat_test_np[~np.isnan(cv_support_feat_test_np)])
                    proto_algo = NearestCentroid(shrink_threshold=shrinkage)
                    proto_algo.fit(cv_support_feat_np, cv_support_target_np)

                    # func_1 = lambda x, y: torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2)
                    dist_func = lambda x, y: -torch.sum(torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2), dim=3)
                    # print(cv_support_feat_test.size())
                    # print(torch.from_numpy(proto_algo.centroids_).to('cuda')[None,...].size())
                    # aux_out = func_1(cv_support_feat_test[None,...], torch.from_numpy(proto_algo.centroids_).to('cuda')[None,...])
                    # print(aux_out.size())
                    output = dist_func(cv_support_feat_test[None,...], torch.from_numpy(proto_algo.centroids_).to('cuda')[None,...])
                    # print(output.size())
                    acc = accuracy(output[0], cv_support_target_test.reshape(-1))

                    # output = proto_algo.predict(cv_support_feat_test_np)
                    # output = torch.from_numpy(output).to('cuda')

                    # print(output)
                    # print(cv_support_target_test)
                    # acc = proto_algo.score(cv_support_feat_test_np, cv_support_target_test_np)
                    
                    # acc = accuracy_cv(output[None,...], cv_support_target_test[None,...])
                    # print(acc)
                    cv_acc.append(acc)
                
                if (sum(cv_acc) / len(cv_acc)) > best_acc:
                    best_acc = sum(cv_acc) / len(cv_acc)
                    best_shrinkage = shrinkage
        
        # print(f"The best shrinkage for this episode is {best_shrinkage} with {best_acc:.2f} accuracy.")
        support_feat = torch.squeeze(support_feat).cpu().detach().numpy()
        support_feat[~np.isfinite(support_feat)] = 0
        support_target = torch.squeeze(support_target).cpu().detach().numpy()
        support_target[~np.isfinite(support_target)] = 0
        query_feat = torch.squeeze(query_feat).cpu().detach().numpy()
        query_feat[~np.isfinite(query_feat)] = 0
        proto_algo = NearestCentroid(shrink_threshold=best_shrinkage)
        proto_algo.fit(support_feat, support_target)
        # output = proto_algo.predict(query_feat)   

        return proto_algo     

        #?===================================================================== 
        # if experimental:
        #     # (1, 1, 1600)
        #     # t, wq, c
        #     query_feat = query_feat.reshape(t, wq, c)
        #     # (1, 25, 1600)
        #     # t, w, c
            
        # else:
        #     # t, wq, c
        #     query_feat = query_feat.reshape(t, way_num * query_num, c)
        #     # t, w, c
        #     # (1, 25, 1600) -> (1, 5, 5, 1600)
        #     support_feat = support_feat.reshape(t, way_num, shot_num, c)
        #     # proto_feat = class prototype 
        #     proto_feat = torch.mean(support_feat, dim=2)

        # return either euclidian distance or cosine similarity
        # return {
        #     # t, wq, 1, c - t, 1, w, c -> t, wq, w
        #     "euclidean": lambda x, y: -torch.sum(
        #         torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
        #         dim=3,
        #     ),
        #     # t, wq, c - t, c, w -> t, wq, w
        #     "cos_sim": lambda x, y: torch.matmul(
        #         F.normalize(x, p=2, dim=-1),
        #         torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
        #         # FEAT did not normalize the query_feat
        #     ),
        # }[mode](query_feat, proto_feat)


class ProtoNetCV(MetricModel):
    def __init__(self, **kwargs):
        super(ProtoNetCV, self).__init__(**kwargs)
        self.proto_layer = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()
        self.evaluator = ProtoCVEvaluator(self, **kwargs)
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

        proto_algo = self.proto_layer(
            query_feat, support_feat, support_target, self.way_num, self.shot_num, self.query_num
        )# .reshape(episode_size * self.way_num * self.query_num, self.way_num)
        # x is the input and y the centroids
        dist_func = lambda x, y: -torch.sum(torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2), dim=3)
        output = dist_func(query_feat, torch.from_numpy(proto_algo.centroids_).to('cuda')[None,...])

        loss = self.loss_func(output[0], query_target.reshape(-1))
        acc = accuracy(output[0], query_target.reshape(-1))
        # print(f"Episode oracle accuracy: {acc:.2f}")

        return output, acc, loss