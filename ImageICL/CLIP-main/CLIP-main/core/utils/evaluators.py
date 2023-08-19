import torch 
import numpy as np

import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression

from utils import create_split_list, create_labels, get_img_emb, get_shuffle

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        self.linear_classifier = LogisticRegression(max_iter=1000)
    
    def eval_episode(self, eval_type, support_feat, support_target, query_feat, query_target, k_fold=5, boot_rounds=30):
        if eval_type == 'oracle':
            outputs = self.oracle_evaluation(support_feat, support_target, query_feat, query_target)
        elif eval_type == 'cross_val':
            outputs = self.cross_val_evaluation(support_feat, support_target, k_fold, ho=False)
        elif eval_type == 'bootstrapping':
            outputs = self.bootstrapping_evaluation(support_feat, support_target, boot_rounds)
        elif eval_type == 'hold_out':
            outputs = self.cross_val_evaluation(support_feat, support_target, k_fold, ho=True)
        else:
            print(f"Evaluation method does not exist.")
            outputs = None

        return outputs
    
    
    def eval_batch(self, config, support_feat, support_target, query_feat=None, query_target=None, k_fold=5, boot_rounds=30):
        eval_results = {}

        for eval_type in config['eval_types']:
            outputs = self.eval_episode(eval_type, support_feat, support_target, query_feat, query_target, k_fold, boot_rounds)
            eval_results[eval_type] = outputs
        
        return eval_results

    def oracle_evaluation(self, support_set, support_target, query_set, query_target):
        intercalated_idx = get_shuffle(self.model.config['test_way'], self.model.config['test_shot'])

        support_set_emb = get_img_emb(support_set, self.model)[intercalated_idx].cpu().detach().numpy()
        support_set_labels = create_labels(support_target, self.model.config['test_way'])[intercalated_idx].cpu().detach().numpy()
        query_set_emb = get_img_emb(query_set, self.model).cpu().detach().numpy()
        query_set_labels = create_labels(query_target, self.model.config['test_way']).cpu().detach().numpy()

        self.linear_classifier.fit(support_set_emb, support_set_labels)
        acc = self.linear_classifier.score(query_set_emb, query_set_labels) * 100
        return {'accuracy' : acc}

    def cross_val_evaluation(self, support_set, support_target, k_fold=5, ho=False):

        n_iters = 1 if ho else k_fold
        intercalated_idx = get_shuffle(self.model.config['test_way'], self.model.config['test_shot'])
        support_set_emb = get_img_emb(support_set, self.model)[intercalated_idx].clone()
        support_set_labels = create_labels(support_target, self.model.config['test_way'])[intercalated_idx].clone()
        
        split_size_list = create_split_list(self.model.config['test_way'] * self.model.config['test_shot'], k_fold)
        support_set_emb = list(torch.split(support_set_emb, split_size_list))
        support_set_labels = list(torch.split(support_set_labels, split_size_list))

        cv_acc = []
        for idx in range(n_iters):
            feat_folds = support_set_emb.copy()
            target_folds = support_set_labels.copy()
            cv_support_feat_test = feat_folds.pop(idx).cpu().detach().numpy()
            cv_support_target_test = target_folds.pop(idx).cpu().detach().numpy()
            cv_support_feat = torch.cat(feat_folds).cpu().detach().numpy()
            cv_support_target = torch.cat(target_folds).cpu().detach().numpy()

            self.linear_classifier.fit(cv_support_feat, cv_support_target)
            acc = self.linear_classifier.score(cv_support_feat_test, cv_support_target_test) * 100
            cv_acc.append(acc)
        
        acc = sum(cv_acc) / len(cv_acc)

        return {'accuracy' : acc}
    
    def bootstrapping_evaluation(self, support_set, support_target, boot_rounds=30):
        rand_gen = np.random.RandomState(seed=self.model.config['boot_seed'])
        class_labels = np.arange(self.model.config['test_way'])
        idx_boot = np.arange(self.model.config['test_way'] * self.model.config['test_shot'])


        cv_acc = []

        boot_feat = get_img_emb(support_set, self.model).clone()
        boot_target = create_labels(support_target, self.model.config['test_way']).clone()

        for round in range(boot_rounds):
            train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
            test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))
            while (len(np.setdiff1d(class_labels, boot_target[train_idx].cpu())) != 0):
                train_idx = torch.from_numpy(rand_gen.choice(idx_boot, size=idx_boot.shape[0], replace=True))
                test_idx = torch.from_numpy(np.setdiff1d(idx_boot, train_idx, assume_unique=False))

            boot_support_feat = boot_feat[train_idx].cpu().detach().numpy()
            boot_support_target = boot_target[train_idx].cpu().detach().numpy()
            boot_support_feat_test = boot_feat[test_idx].cpu().detach().numpy()
            boot_support_target_test = boot_target[test_idx].cpu().detach().numpy()
            self.linear_classifier.fit(boot_support_feat, boot_support_target)
            acc = self.linear_classifier.score(boot_support_feat_test, boot_support_target_test) * 100
            cv_acc.append(acc)
        
        acc = sum(cv_acc) / len(cv_acc)
        return {'accuracy' : acc}