# -*- coding: utf-8 -*-
import datetime
import logging
import os
import builtins
from logging import getLogger
from time import time
import csv

import pandas as pd

import torch
import yaml
from torch import nn
import torch.distributed as dist
import optuna

from queue import Queue
from core import Trainer

from core.utils import (
    ModelType,
    SaveType,
)

class HPOExperiment(object):

    def __init__(self, rank, config):
        # super(HPOExperiment, self).__init__(rank, config)
        self.config = config
        self.rank = rank

        self.config['eval_types'] = ['oracle']
        # self.config['epoch'] = int(self.config['epoch'] * 0.15)
    
    def save_hpo_logs(self, result_dir, ckpt_dir, best_val_acc, filename='hpo_logs.csv'):
        mode = 'a'
        if not os.path.isfile(os.path.join(result_dir, filename)):
            mode = 'w'
        
        with open(os.path.join(result_dir, filename), mode) as logs:
            writer = csv.writer(logs)
            if mode == 'w':
                writer.writerow(['result_dir', 'best_val_acc'])
            writer.writerow([ckpt_dir, best_val_acc])

    
    def objective(self, trial):
        #* define hyperparameter values to optimizer here
        #* it will depend on model type however (some of them) -> check classifier name
        # defining range of hyperparameters to optimize
        current_trial = trial.number
        print("Current trial: " + str(current_trial))
        print(self.rank)
        print(self.config)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        self.config['batch_size'] = batch_size
        self.config['optimizer']['kwargs']['lr'] = lr
        if self.config['classifier']['name'] == 'MAML':
            self.config['classifier']['kwargs']['inner_param']['lr'] = trial.suggest_float("lr_inner_maml", 1e-4, 0.1, log=True)


        print(self.config)
        trainer = Trainer(self.rank, self.config)

        """
        The normal train loop: train-val-test and save model when val-acc increases.
        """
        experiment_begin = time()
        for epoch_idx in range(trainer.from_epoch + 1, trainer.config["epoch"]):
            if trainer.distribute and trainer.model_type == ModelType.FINETUNING:
                trainer.train_loader[0].sampler.set_epoch(epoch_idx)
            print("============ Train on the train set ============")
            print("learning rate: {}".format(trainer.scheduler.get_last_lr()))
            train_acc = trainer._train(epoch_idx)
            print(" * Acc@1 {:.3f} ".format(train_acc))
            if ((epoch_idx + 1) % trainer.val_per_epoch) == 0:
                print("============ Validation on the val set ============")
                val_acc = trainer._validate(epoch_idx, is_test=False)
                trainer.val_acc = val_acc
                print(
                    " * Acc@1 {:.3f} Best acc {:.3f}".format(val_acc, trainer.best_val_acc)
                )
            time_scheduler = trainer._cal_time_scheduler(experiment_begin, epoch_idx)
            print(" * Time: {}".format(time_scheduler))
            trainer.scheduler.step()

            trial.report(trainer.val_acc, epoch_idx)
            # handle pruning based on the intermediate value
            if trial.should_prune():
                print("PRUNING")
                raise optuna.exceptions.TrialPruned()

            if trainer.rank == 0:
                if ((epoch_idx + 1) % trainer.val_per_epoch) == 0:
                    if val_acc > trainer.best_val_acc:
                        trainer.best_val_acc = val_acc
                        trainer.best_epoch = epoch_idx
                        # trainer.best_test_acc = test_acc
                #         trainer._save_model(epoch_idx, SaveType.BEST)

                #     if epoch_idx != 0 and epoch_idx % trainer.config["save_interval"] == 0:
                #         trainer._save_model(epoch_idx, SaveType.NORMAL)

                # trainer._save_model(epoch_idx, SaveType.LAST)
            
            if (epoch_idx - trainer.best_epoch) > trainer.early_stopping_thresh:
                print(f"Early stop at epoch {epoch_idx}.")
                break

        if trainer.rank == 0:
            print(
                "End of experiment, took {}".format(
                    str(datetime.timedelta(seconds=int(time() - experiment_begin)))
                )
            )
            print("Result DIR: {}".format(trainer.result_path))

        if trainer.writer is not None:
            trainer.writer.close()
            if trainer.distribute:
                dist.barrier()
        elif trainer.distribute:
            dist.barrier()
        
        self.save_hpo_logs(trainer.config['result_root'], trainer.result_path, trainer.best_val_acc)

        print(trainer.best_val_acc)
        return trainer.best_val_acc
