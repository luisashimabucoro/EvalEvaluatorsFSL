# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch
from torch import nn
import math
import time
import numpy as np
from time import time, sleep

from core.utils import accuracy, get_shuffle, check_gpu_memory_usage, create_split_list, get_permutated_shuffle_combinations

class AbstractEvaluator(object):
    """
    Init:
        - type of model
        - episode size, way, shot etc
        - type of evaluation (cross-val, regular etc)
    
    Basic methods:
        - evaluate episode
        - evaluate batch

    """
    def __init__(self, model, **kwargs):
        super(AbstractEvaluator, self).__init__()

        self.model = model

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    
    def eval_episode(self, eval_type, episode_size, support_feat, support_target, query_feat, query_target, k_fold=None, boot_rounds=30):
        if eval_type == 'oracle':
            outputs = self.oracle_evaluation(episode_size, support_feat, support_target, query_feat, query_target)
        elif eval_type == 'cross_val':
            outputs = self.cross_val_evaluation(episode_size, support_feat, support_target, query_feat, k_fold, ho=False)
        elif eval_type == 'improved_cross_val':
            outputs = self.improved_cross_val_evaluation(episode_size, support_feat, support_target, query_feat, k_fold, ho=False)
        elif eval_type == 'bootstrapping':
            outputs = self.bootstrapping_evaluation(episode_size, support_feat, support_target, query_feat, boot_rounds)
        elif eval_type == 'hold_out':
            outputs = self.cross_val_evaluation(episode_size, support_feat, support_target, query_feat, k_fold, ho=True)
        else:
            print(f"Evaluation method does not exist.")
            outputs = None

        return outputs
    
    
    def eval_batch(self, episode_size, support_feat, support_target, query_feat, query_target, k_fold=None, boot_rounds=100):
        eval_results = {}

        for eval_type in self.model.eval_types:
            outputs = self.eval_episode(eval_type, episode_size, support_feat, support_target, query_feat, query_target, k_fold, boot_rounds)
            eval_results[eval_type] = outputs
        
        return eval_results
    
    @abstractmethod
    def oracle_evaluation(self, *args, **kwargs):
        return
    
    @abstractmethod
    def cross_val_evaluation(self, *args, **kwargs):
        return

    @abstractmethod
    def bootstrapping_evaluation(self, *args, **kwargs):
        return



class BaselineEvaluator(AbstractEvaluator):
    def __init__(self, model, **kwargs):
        super(BaselineEvaluator, self).__init__(model, **kwargs)

    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        output_list = []
        for episode_idx in range(episode_size):
            _, output = self.model.set_forward_adaptation_og(support_feat[episode_idx], support_target[episode_idx], query_feat[episode_idx])
            # output_list.append(torch.from_numpy(output))
            output_list.append(output)

        # output = torch.cat(output_list, dim=-1).to(self.model.device)
        output = torch.cat(output_list, dim=-1)

        loss = self.model.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        # print(f"Oracle Stats - Loss: {loss}\t Acc: {acc}")
        return {'output' : output, 'loss' : loss, 'accuracy' : acc}
    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold=5, ho=False):
        cv_loss = []
        cv_acc = []
        output_list = []
        # print(self.model.way_num)
        # print(self.model.shot_num)
        n_iters = 1 if ho else k_fold

        for episode_idx in range(episode_size):
            # evenly spacing support set
            # eg: 0,  5, 10, 15, 20,  1,  6, 11, 16, 21,  2,  7, 12, 17, 22,  3,  8, 13, 18, 23,  4,  9, 14, 19, 24
            shuffle_idx = get_shuffle(self.model.way_num, self.model.shot_num)
            shuffled_feat = support_feat[episode_idx][shuffle_idx].clone()
            shuffled_target = support_target[episode_idx][shuffle_idx].clone()
            # creating folds
            split_size_list = create_split_list(self.model.way_num * self.model.shot_num, k_fold)
            shuffled_feat = list(torch.split(shuffled_feat, split_size_list))
            shuffled_target = list(torch.split(shuffled_target, split_size_list))

            for idx in range(n_iters):
                # dividing train folds from test fold
                feat_folds = shuffled_feat.copy()
                target_folds = shuffled_target.copy()
                cv_support_feat_test = feat_folds.pop(idx)
                cv_support_target_test = target_folds.pop(idx)
                cv_support_feat = torch.cat(feat_folds)
                cv_support_target = torch.cat(target_folds)
                
                _, output = self.model.set_forward_adaptation_og(cv_support_feat, cv_support_target, cv_support_feat_test)
                # classifier, _ = self.model.set_forward_adaptation(cv_support_feat, cv_support_target, query_feat[0])
                # output = classifier.predict_proba(cv_support_feat_test.cpu())
                # output = torch.from_numpy(output).to(self.model.device)

                experimental_loss = self.model.loss_func(output, cv_support_target_test)
                # print(output.size())
                # print(output)
                # print(cv_support_target_test.size())
                # print(cv_support_target_test)
                acc = accuracy(output, cv_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
                # print(f"Fold {idx+1} completed!")
        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc}

    def bootstrapping_evaluation(self, episode_size, support_feat, support_target, query_feat, boot_rounds=30):
        rand_gen = np.random.RandomState(seed=self.model.boot_seed)
        class_labels = np.arange(self.model.way_num)
        idx_boot = np.arange(self.model.way_num * self.model.shot_num)
        episode_dict = {}

        cv_loss = []
        cv_acc = []
        output_list = []

        for episode_idx in range(episode_size):
            boot_feat = support_feat[episode_idx].clone()
            boot_target = support_target[episode_idx].clone()

            for round in range(boot_rounds):
                train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                while (len(np.setdiff1d(class_labels, boot_target[train_idx].cpu())) != 0):
                    train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                    test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                boot_support_feat = boot_feat[train_idx]
                boot_support_target = boot_target[train_idx]
                boot_support_feat_test = boot_feat[test_idx]
                boot_support_target_test = boot_target[test_idx]
                
                _, output = self.model.set_forward_adaptation_og(boot_support_feat, boot_support_target, boot_support_feat_test)
                # classifier, _ = self.model.set_forward_adaptation(boot_support_feat, boot_support_target, query_feat[0])
                # output = classifier.predict_proba(boot_support_feat_test.cpu())
                # output = torch.from_numpy(output).to(self.model.device)

                experimental_loss = self.model.loss_func(output, boot_support_target_test)

                acc = accuracy(output, boot_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
                episode_dict[round+1] = sum(cv_acc) / len(cv_acc)
        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        # print(f"Boots Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc, 'episode_dict' : episode_dict}


class BaselinePPEvaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(BaselinePPEvaluator, self).__init__(model, **kwargs)

    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        output_list = []
        for i in range(episode_size):
            _, output = self.model.set_forward_adaptation_og(
                support_feat[i], support_target[i], query_feat[i]
            )
            output_list.append(output)

        output = torch.cat(output_list, dim=0).to(self.model.device)
        # output = torch.cat(output_list, dim=0)
        # print(f"Output shape: {output.size()}\t Target shape: {torch.squeeze(query_target).size()}")
        # print(f"Output shape: {output.size()}\t Target shape: {query_target.size()}")
        loss = self.model.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        # print(f"Oracle Stats - Loss: {loss}\t Acc: {acc}")

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}
    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold=5, ho=False):
        cv_loss = []
        cv_acc = []
        output_list = []
        # print(self.model.way_num)
        # print(self.model.shot_num)
        n_iters = 1 if ho else k_fold

        for episode_idx in range(episode_size):
            # evenly spacing support set
            # eg: 0,  5, 10, 15, 20,  1,  6, 11, 16, 21,  2,  7, 12, 17, 22,  3,  8, 13, 18, 23,  4,  9, 14, 19, 24
            shuffle_idx = get_shuffle(self.model.way_num, self.model.shot_num)
            shuffled_feat = support_feat[episode_idx][shuffle_idx].clone()
            shuffled_target = support_target[episode_idx][shuffle_idx].clone()
            # creating folds
            split_size_list = create_split_list(self.model.way_num * self.model.shot_num, k_fold)
            shuffled_feat = list(torch.split(shuffled_feat, split_size_list))
            shuffled_target = list(torch.split(shuffled_target, split_size_list))

            for idx in range(n_iters):
                # dividing train folds from test fold
                feat_folds = shuffled_feat.copy()
                target_folds = shuffled_target.copy()
                cv_support_feat_test = feat_folds.pop(idx)
                cv_support_target_test = target_folds.pop(idx)
                cv_support_feat = torch.cat(feat_folds)
                cv_support_target = torch.cat(target_folds)
                
                _, output = self.model.set_forward_adaptation_og(cv_support_feat, cv_support_target, cv_support_feat_test)
                output = output.to(self.model.device)

                experimental_loss = self.model.loss_func(output, cv_support_target_test)
                acc = accuracy(output, cv_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc}

    def bootstrapping_evaluation(self, episode_size, support_feat, support_target, query_feat, boot_rounds=30):
        rand_gen = np.random.RandomState(seed=42)
        class_labels = np.arange(self.model.way_num)
        idx_boot = np.arange(self.model.way_num * self.model.shot_num)
        episode_dict = {}

        cv_loss = []
        cv_acc = []
        output_list = []

        for episode_idx in range(episode_size):
            boot_feat = support_feat[episode_idx].clone()
            boot_target = support_target[episode_idx].clone()

            for round in range(boot_rounds):
                train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                while (len(np.setdiff1d(class_labels, boot_target[train_idx].cpu())) != 0):
                    train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                    test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                boot_support_feat = boot_feat[train_idx]
                boot_support_target = boot_target[train_idx]
                boot_support_feat_test = boot_feat[test_idx]
                boot_support_target_test = boot_target[test_idx]
                
                _, output = self.model.set_forward_adaptation_og(boot_support_feat, boot_support_target, boot_support_feat_test)
                output = output.to(self.model.device)

                experimental_loss = self.model.loss_func(output, boot_support_target_test)
                acc = accuracy(output, boot_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
                episode_dict[round+1] = sum(cv_acc) / len(cv_acc)
        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc, 'episode_dict' : episode_dict}



class ProtoEvaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(ProtoEvaluator, self).__init__(model, **kwargs)

    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        output = self.model.proto_layer(query_feat, support_feat, support_target, self.model.way_num, self.model.shot_num, self.model.query_num
                                        ).reshape(episode_size * self.model.way_num * self.model.query_num, self.model.way_num)
        
        loss = self.model.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        # print(f"Oracle Stats - Loss: {loss}\t Acc: {acc}")

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}

    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold=5, ho=False):
        # return
        cv_loss = []
        cv_acc = []
        output_list = []

        n_iters = 1 if ho else k_fold
        # print(f"Test way: {self.model.way_num}\t Test shot: {self.model.shot_num}")
        episode_shuffle_idx = get_shuffle(self.model.way_num, self.model.shot_num)
        split_size_list = create_split_list(self.model.way_num * self.model.shot_num, k_fold)
        episode_shuffle_idx = list(torch.split(episode_shuffle_idx, split_size_list))

        for episode_idx in range(episode_size):
            cv_support_feat = support_feat[episode_idx].clone()
            cv_support_target = support_target[episode_idx].clone()
            
            for idx in range(n_iters):
                shuffle_idx = episode_shuffle_idx.copy()
                test_idx = shuffle_idx.pop(idx)
                train_idx, _ = torch.sort(torch.cat(shuffle_idx))
                cv_support_feat = support_feat[episode_idx].clone()
                cv_support_target = support_target[episode_idx].clone()
                cv_support_feat_test = cv_support_feat[test_idx]
                cv_support_target_test = cv_support_target[test_idx]
                cv_support_feat = cv_support_feat[train_idx]
                cv_support_target = cv_support_target[train_idx]

                # defining number of samples per class used to determine the class prototypes
                train_samples_per_class = {class_num : self.model.shot_num for class_num in range(self.model.way_num)}
                for idx in cv_support_target_test:
                    train_samples_per_class[int(idx)] -= 1
                
                # (batch, n_samples, feat_size)
                output = self.model.proto_layer(
                    cv_support_feat_test[None, ...], cv_support_feat[None, ...], cv_support_target, self.model.way_num, self.model.shot_num, self.model.query_num, experimental=True, samples_per_class=train_samples_per_class
                )
                # print(k_fold)
                # print(output.size())
                if output.size() == torch.Size([1, 1, self.model.way_num]):
                    output = torch.squeeze(output, 0)
                else: 
                    output = torch.squeeze(output)

                # experimental_loss = self.model.loss_func(output, cv_support_target_test[None, ...])
                experimental_loss = self.model.loss_func(output, cv_support_target_test)
                acc = accuracy(output, cv_support_target_test.reshape(-1))

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)

        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc}
    
    def bootstrapping_evaluation(self, episode_size, support_feat, support_target, query_feat, boot_rounds=30):
        rand_gen = np.random.RandomState(seed=42)
        class_labels = np.arange(self.model.way_num)
        idx_boot = np.arange(self.model.way_num * self.model.shot_num)

        cv_loss = []
        cv_acc = []
        output_list = []
        episode_dict = {}

        for episode_idx in range(episode_size):
            boot_feat = support_feat[episode_idx].clone()
            boot_target = support_target[episode_idx].clone()

            for round in range(boot_rounds):
                train_idx, _ = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True)).sort()
                test_idx, _ = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False)).sort()
                while (len(np.setdiff1d(class_labels, boot_target[train_idx].cpu())) != 0):
                    train_idx, _ = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True)).sort()
                    test_idx, _ = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False)).sort()

                boot_support_feat = boot_feat[train_idx]
                boot_support_target = boot_target[train_idx]
                boot_support_feat_test = boot_feat[test_idx]
                boot_support_target_test = boot_target[test_idx]
                # defining number of samples per class used to determine the class prototypes
                train_samples_per_class = {class_num : 0 for class_num in range(self.model.way_num)}
                for idx in boot_support_target:
                    train_samples_per_class[int(idx)] += 1
                # print(boot_support_target)
                # print(train_idx)
                # print(train_samples_per_class)
                # sleep(20)
                
                # (batch, n_samples, feat_size)
                output = self.model.proto_layer(
                    boot_support_feat_test[None, ...], boot_support_feat[None, ...], boot_support_target, self.model.way_num, self.model.shot_num, self.model.query_num, experimental=True, samples_per_class=train_samples_per_class
                )
                # print(k_fold)
                # print(output.size())
                if output.size() == torch.Size([1, 1, self.model.way_num]):
                    output = torch.squeeze(output, 0)
                else: 
                    output = torch.squeeze(output)

                # experimental_loss = self.model.loss_func(output, cv_support_target_test[None, ...])
                # print(output.size(), boot_support_target_test.size())
                # print(output, boot_support_target_test)
                experimental_loss = self.model.loss_func(output, boot_support_target_test)
                acc = accuracy(output, boot_support_target_test.reshape(-1))
                # print(f"Loss: {experimental_loss}")
                # print(f"Acc: {acc}")

                if not math.isnan(experimental_loss):
                    cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
                episode_dict[round+1] = sum(cv_acc) / len(cv_acc)

        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)
        # print(f"Boot Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc, 'episode_dict' : episode_dict}

class ProtoCVEvaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(ProtoCVEvaluator, self).__init__(model, **kwargs)

    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        proto_algo = self.model.proto_layer(query_feat, support_feat, support_target, self.model.way_num, self.model.shot_num, self.model.query_num, experimental=True)
        dist_func = lambda x, y: -torch.sum(torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2), dim=3)
        output = dist_func(query_feat, torch.from_numpy(proto_algo.centroids_).to('cuda')[None,...])

        loss = self.model.loss_func(output[0], query_target.reshape(-1))
        acc = accuracy(output[0], query_target.reshape(-1))

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}

class MAMLEvaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(MAMLEvaluator, self).__init__(model, **kwargs)
    
    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        # (ep_size, batch_size, channels, height, width) ?
        _, _, c, h, w = support_feat.size()

        output_list = []
        # iterating through episodes
        for i in range(episode_size):
            episode_support_image = support_feat[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            episode_query_image = query_feat[i].contiguous().reshape(-1, c, h, w)
            # episode_query_target = query_target[i].reshape(-1)

            # fine-tuning model with support set
            # l = self.model.set_forward_adaptation(episode_support_image, episode_support_target)
            l = self.model.set_forward_adaptation(episode_support_image, episode_support_target)

            # testing model on the query set
            output = self.model.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.model.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        # print(f"Oracle Stats - Loss: {loss}\t Acc: {acc}")

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}
    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold=5, ho=False):
        episode_shuffle_idx = get_shuffle(self.model.way_num, self.model.shot_num)
        # query_shuffle_idx = get_shuffle(self.model.way_num, self.model.query_num)
        # print(query_shuffle_idx)
        split_size_list = create_split_list(self.model.way_num * self.model.shot_num, k_fold)
        episode_shuffle_idx = list(torch.split(episode_shuffle_idx, split_size_list))

        n_iters = 1 if ho else k_fold
        _, _, c, h, w = support_feat.size()

        cv_loss = []
        cv_acc = []
        output_list = []
        # iterating through episodes
        for episode_idx in range(episode_size):
            cv_support_feat = support_feat[episode_idx].clone()
            cv_support_target = support_target[episode_idx].clone()

            # (ep_size, batch_size, channels, height, width)
            # (support_samples, channels, height, width)
            for idx in range(n_iters):
                with torch.no_grad():
                    shuffle_idx = episode_shuffle_idx.copy()
                    test_idx = shuffle_idx.pop(idx)
                    train_idx = torch.cat(shuffle_idx)


                    cv_support_feat = support_feat[episode_idx].contiguous().reshape(-1, c, h, w)
                    cv_support_target = support_target[episode_idx].reshape(-1)
                    cv_support_feat_test = cv_support_feat[test_idx]
                    cv_support_target_test = cv_support_target[test_idx]
                cv_support_feat = cv_support_feat[train_idx]
                cv_support_target = cv_support_target[train_idx]

                # cv_support_feat = support_feat[episode_idx].contiguous().reshape(-1, c, h, w)
                # cv_support_target = support_target[episode_idx].reshape(-1)

                # fine-tuning model with support set
                l = self.model.set_forward_adaptation(cv_support_feat.contiguous(), cv_support_target.reshape(-1))

                with torch.no_grad():
                    # testing model on the query set
                    # training_shuffle_idx = torch.randperm(cv_support_feat_test.size(0))
                    # print(output, cv_support_target_test.contiguous().view(-1))
                    output = self.model.forward_output(cv_support_feat_test.contiguous())
                    loss = self.model.loss_func(output, cv_support_target_test.contiguous().view(-1))
                    acc = accuracy(output, cv_support_target_test.contiguous().view(-1))
                    output_list.append(output)
                    cv_loss.append(loss)
                    cv_acc.append(acc)
                
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        # print(f"CV Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output_list, 'loss' : cv_loss, 'accuracy' : cv_acc}

    def bootstrapping_evaluation(self, episode_size, support_feat, support_target, query_feat, boot_rounds=30):
        class_labels = np.arange(self.model.way_num)
        rand_gen = np.random.RandomState(seed=42)
        idx_boot = np.arange(self.model.way_num * self.model.shot_num)
        episode_dict = {}


        _, _, c, h, w = support_feat.size()

        cv_loss = []
        cv_acc = []
        output_list = []

        for episode_idx in range(episode_size):
            boot_feat = support_feat[episode_idx].clone().contiguous().reshape(-1, c, h, w)
            boot_target = support_target[episode_idx].clone().contiguous().reshape(-1)

            for round in range(boot_rounds):
                with torch.no_grad():
                    # print(round)
                    train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                    test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                    # print('Before')
                    # check_gpu_memory_usage([0])
                    factor = np.setdiff1d(class_labels, boot_target[train_idx].cpu())
                    while (len(factor) != 0):
                        train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                        test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                        factor = np.setdiff1d(class_labels, boot_target[train_idx].cpu())
                    # print('After')
                    # check_gpu_memory_usage([0])
                        # torch.cuda.empty_cache()
                    # del factor
                    boot_support_feat_test = boot_feat[test_idx]
                    boot_support_target_test = boot_target[test_idx]
                boot_support_feat = boot_feat[train_idx]
                boot_support_target = boot_target[train_idx]

                    # fine-tuning model with support set
                l = self.model.set_forward_adaptation(boot_support_feat.contiguous(), boot_support_target.reshape(-1))
                del l 
                with torch.no_grad():
                    # testing model on the query set
                    training_shuffle_idx = torch.randperm(boot_support_feat_test.size(0))
                    output = self.model.forward_output(boot_support_feat_test[training_shuffle_idx].contiguous())
                    # print(output, cv_support_target_test[training_shuffle_idx].contiguous().view(-1))
                    loss = self.model.loss_func(output, boot_support_target_test[training_shuffle_idx].contiguous().view(-1))
                    acc = accuracy(output, boot_support_target_test[training_shuffle_idx].contiguous().view(-1))
                    # output_list.append(output)
                    output_list.append(0)
                    cv_loss.append(loss)
                    cv_acc.append(acc)
                    episode_dict[round+1] = sum(cv_acc) / len(cv_acc)

                # del boot_support_feat
                # del boot_support_target
                # del boot_support_feat_test
                # del boot_support_target_test
                # del output
                # del training_shuffle_idx
                # del train_idx
                # del test_idx
                # torch.cuda.empty_cache()
            
            # del boot_feat
            # del boot_target
            # torch.cuda.empty_cache()
                
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        # output = torch.cat(output_list, dim=0)
        output = []
        print(episode_dict)
        # print(f"CV Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output_list, 'loss' : cv_loss, 'accuracy' : cv_acc, 'episode_dict' : episode_dict}

class MAMLCVEvaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(MAMLCVEvaluator, self).__init__(model, **kwargs)
    
    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        # (ep_size, batch_size, channels, height, width) ?
        _, _, c, h, w = support_feat.size()

        output_list = []
        # iterating through episodes
        for i in range(episode_size):
            episode_query_image = query_feat[i].contiguous().reshape(-1, c, h, w)
            # episode_query_target = query_target[i].reshape(-1)

            # fine-tuning model with support set
            # l = self.model.set_forward_adaptation(episode_support_image, episode_support_target)
            l = self.model.set_forward_adaptation_cv(support_feat, support_target)

            # testing model on the query set
            output = self.model.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.model.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        # print(f"Oracle Stats - Loss: {loss}\t Acc: {acc}")

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}

class R2D2Evaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(R2D2Evaluator, self).__init__(model, **kwargs)
    
    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        output, weight = self.model.classifier(
            self.model.way_num, self.model.shot_num, query_feat, support_feat, support_target
        )

        output = output.contiguous().reshape(-1, self.model.way_num)
        loss = self.model.loss_func(output, query_target.contiguous().reshape(-1))
        acc = accuracy(output.squeeze(), query_target.contiguous().reshape(-1))

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}
    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold=5, ho=False):
        cv_loss = []
        cv_acc = []
        output_list = []
        # print(self.model.way_num)
        # print(self.model.shot_num)
        n_iters = 1 if ho else k_fold

        for episode_idx in range(episode_size):
            # evenly spacing support set
            # eg: 0,  5, 10, 15, 20,  1,  6, 11, 16, 21,  2,  7, 12, 17, 22,  3,  8, 13, 18, 23,  4,  9, 14, 19, 24
            shuffle_idx = get_shuffle(self.model.way_num, self.model.shot_num)
            shuffled_feat = support_feat[episode_idx][shuffle_idx].clone()
            shuffled_target = support_target[episode_idx][shuffle_idx].clone()
            # creating folds
            split_size_list = create_split_list(self.model.way_num * self.model.shot_num, k_fold)
            shuffled_feat = list(torch.split(shuffled_feat, split_size_list))
            shuffled_target = list(torch.split(shuffled_target, split_size_list))

            for idx in range(n_iters):
                # dividing train folds from test fold
                feat_folds = shuffled_feat.copy()
                target_folds = shuffled_target.copy()
                cv_support_feat_test = feat_folds.pop(idx)
                cv_support_target_test = target_folds.pop(idx)
                cv_support_feat = torch.cat(feat_folds)
                cv_support_target = torch.cat(target_folds)
                
                if cv_support_feat_test.dim() != 3:
                    cv_support_feat_test = cv_support_feat_test[None, ...]
                if cv_support_feat.dim() != 3:
                    cv_support_feat = cv_support_feat[None, ...]

                output, weight = self.model.classifier(
                self.model.way_num, self.model.shot_num, cv_support_feat_test, cv_support_feat, cv_support_target)

                if output.size(0) == 1:
                    output = torch.squeeze(output, 0)

                experimental_loss = self.model.loss_func(output, cv_support_target_test)
                acc = accuracy(output, cv_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)
        # print(f"CV Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc}
    
    def improved_cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold=5, ho=False):
        cv_loss = []
        cv_acc = []
        output_list = []
        # print(self.model.way_num)
        # print(self.model.shot_num)

        for episode_idx in range(episode_size):
            # evenly spacing support set
            # eg: 0,  5, 10, 15, 20,  1,  6, 11, 16, 21,  2,  7, 12, 17, 22,  3,  8, 13, 18, 23,  4,  9, 14, 19, 24
            support_idx, testing_splits = get_permutated_shuffle_combinations(self.model.way_num, self.model.shot_num)
            print(testing_splits)
            sleep(10)

            for test_idx in testing_splits:
                train_idx = np.setdiff1d(support_idx, test_idx)
                cv_support_feat = support_feat[torch.from_numpy(train_idx)]
                cv_support_target = support_target[torch.from_numpy(train_idx)]
                cv_support_feat_test = support_feat[torch.from_numpy(test_idx)]
                cv_support_target_test = support_target[torch.from_numpy(test_idx)]
                
                if cv_support_feat_test.dim() != 3:
                    cv_support_feat_test = cv_support_feat_test[None, ...]
                if cv_support_feat.dim() != 3:
                    cv_support_feat = cv_support_feat[None, ...]

                output, weight = self.model.classifier(
                self.model.way_num, self.model.shot_num, cv_support_feat_test, cv_support_feat, cv_support_target)

                if output.size(0) == 1:
                    output = torch.squeeze(output, 0)

                experimental_loss = self.model.loss_func(output, cv_support_target_test)
                acc = accuracy(output, cv_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)
        # print(f"CV Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc}

    def bootstrapping_evaluation(self, episode_size, support_feat, support_target, query_feat, boot_rounds=30):
        class_labels = np.arange(self.model.way_num)
        rand_gen = np.random.RandomState(seed=42)
        idx_boot = np.arange(self.model.way_num * self.model.shot_num)

        cv_loss = []
        cv_acc = []
        output_list = []
        episode_dict = {}

        for episode_idx in range(episode_size):
            boot_feat = support_feat[episode_idx].clone()
            boot_target = support_target[episode_idx].clone()

            for round in range(boot_rounds):
                train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                while (len(np.setdiff1d(class_labels, boot_target[train_idx].cpu())) != 0):
                    train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                    test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
                boot_support_feat = boot_feat[train_idx]
                boot_support_target = boot_target[train_idx]
                boot_support_feat_test = boot_feat[test_idx]
                boot_support_target_test = boot_target[test_idx]

                if boot_support_feat_test.dim() != 3:
                    boot_support_feat_test = boot_support_feat_test[None, ...]
                if boot_support_feat.dim() != 3:
                    boot_support_feat = boot_support_feat[None, ...]

                output, weight = self.model.classifier(
                self.model.way_num, self.model.shot_num, boot_support_feat_test, boot_support_feat, boot_support_target)

                if output.size(0) == 1:
                    output = torch.squeeze(output, 0)

                experimental_loss = self.model.loss_func(output, boot_support_target_test)
                acc = accuracy(output, boot_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
                episode_dict[round+1] = sum(cv_acc) / len(cv_acc)

        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)
        # print(f"CV Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc, 'episode_dict' : episode_dict}