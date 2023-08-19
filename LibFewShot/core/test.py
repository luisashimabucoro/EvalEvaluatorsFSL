# -*- coding: utf-8 -*-
import datetime
import logging
import os
import builtins
from logging import getLogger
from time import time, sleep

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import yaml
from torch import nn
import torch.distributed as dist
import optuna

import shutil
import csv

from queue import Queue
import core.model as arch
from core.data import get_dataloader, DataLoaderCrossDomain, create_datasets, MetaAlbumDataset
from core.utils import (
    init_logger_config,
    prepare_device,
    init_seed,
    create_dirs,
    AverageMeter,
    count_parameters,
    ModelType,
    TensorboardWriter,
    mean_confidence_interval,
    get_instance,
    get_local_time
)

import pandas as pd

class Test(object):
    """
    The tester.

    Build a tester from config dict, set up model from a saved checkpoint, etc. Test and log.
    """

    def __init__(self, rank, config, result_path=None):
        self.rank = rank
        if (config.get('fixed_classes') and (config.get('fixed_classes') != 0)):
            config['test_shot'] = config['fixed_classes'] // config['test_way']
            # config['way_num'] = config['test_way']
        if config['experiment_dir'] == 'bootstrapping_analysis':
            config['boot_rounds'] = 50
        if not config.get('boot_seed'):
            config['boot_seed'] = 42
        self.config = config
        if not config.get('cross_domain'):
            config['test_episode'] = 600
            config['episode_size'] = 1
        # self.config['eval_types'] = ["oracle", "cross_val", "bootstrapping"]
        self.config["rank"] = rank
        self._init_result_root()
        self.result_path = result_path
        self.distribute = self.config["n_gpu"] > 1
        self.viz_path, self.state_dict_path = self._init_files(config)
        self.logger = self._init_logger()
        self.device, self.list_ids = self._init_device(rank, config)
        self.writer = self._init_writer(self.viz_path)
        self.test_meter = self._init_meter()
        print(config)
        if (config.get('fixed_classes') and (config.get('fixed_classes') != 0)):
            config['way_num'] = config['test_way']
            config['shot_num'] = config['test_shot']
        
        if config.get('cross_domain'):
            if config['classifier']['name'] in ['Baseline', 'BaselinePlus', 'BaselinePlusCV']:
                if config['val_datasets'][0] == '44298/DOG_Mini':
                    config['classifier']['kwargs']['num_class'] = 1658
                elif config['val_datasets'][0] == '44285/BRD_Mini':
                    config['classifier']['kwargs']['num_class'] = 1273
        self.model, self.model_type = self._init_model(config)

        if self.config.get('stability_test') == 'true':
            original_shot = self.config['test_shot']
            config['test_shot'] = original_shot + 1
            self.test_loader = self._init_dataloader(config, config.get('cross_domain'))
            # print(f"New shot for stability test: {self.config['test_shot']}")
            config['test_shot'] = original_shot
        else:
            self.test_loader = self._init_dataloader(config, config.get('cross_domain'))
        # self.intermidiate_stats = {eval_type : {'test_loss' : [], 'test_accuracy' : []} for eval_type in self.config['eval_types']}
        self.intermidiate_stats = {}
        if self.config.get('cross_domain'):
           print(f"{self.config['val_datasets'][0][-8:]}.csv")
           print(self.config['train_datasets'])
           print(self.config['val_datasets'])
           print(self.config['test_datasets'])

    def test_loop(self):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """
        total_accuracy = 0.0
        total_h = np.zeros(self.config["test_epoch"])
        total_accuracy_vector = []

        for eval_type in self.config['eval_types']:
            self.model.eval_types = [eval_type]
            if eval_type == 'cross_val':
                for fold in self.config['test_folds']:
                    if self.config.get('stability_test') == 'true':
                        original_shot = self.config['test_shot']
                        self.config['test_shot'] = original_shot + 1
                        self.test_loader = self._init_dataloader(self.config, self.config.get('cross_domain'))
                        # print(f"New shot for stability test: {self.config['test_shot']}")
                        self.config['test_shot'] = original_shot
                    else:
                        self.test_loader = self._init_dataloader(self.config, self.config.get('cross_domain'))
                    # setting loo cross validation value before testing
                    fold_value = (self.config['test_way'] * self.config['test_shot']) if (int(fold) == 1000) else int(fold)
                    self.model.k_fold = fold_value
                    self.intermidiate_stats[f'{eval_type}-{fold}fold'] = {'test_loss' : [], 'test_accuracy' : []}
                    for epoch_idx in range(self.config["test_epoch"]):
                        print("============ Testing on the test set ============")
                        print(f'Evaluation: {eval_type} with {self.model.k_fold} folds') 
                        if self.config.get('cross_domain'):
                            _, accuracies = self._validate_md(epoch_idx, f'{eval_type}-{fold}fold')
                        else:   
                            _, accuracies = self._validate(epoch_idx, f'{eval_type}-{fold}fold')
                        test_accuracy, h = mean_confidence_interval(accuracies)
                        print("Test Accuracy: {:.3f}\t h: {:.3f}".format(test_accuracy, h))
                        total_accuracy += test_accuracy
                        total_accuracy_vector.extend(accuracies)
                        total_h[epoch_idx] = h
            else:
                # reinitiating dataloader so test episodes are the same for all testing methods
                if self.config.get('stability_test') == 'true':
                    original_shot = self.config['test_shot']
                    self.config['test_shot'] = original_shot + 1
                    self.test_loader = self._init_dataloader(self.config, self.config.get('cross_domain'))
                    # print(f"New shot for stability test: {self.config['test_shot']}")
                    self.config['test_shot'] = original_shot
                else:
                    self.test_loader = self._init_dataloader(self.config, self.config.get('cross_domain'))
                self.intermidiate_stats[f'{eval_type}'] = {'test_loss' : [], 'test_accuracy' : []}
                for epoch_idx in range(self.config["test_epoch"]):
                    print("============ Testing on the test set ============")
                    print(f'Evaluation: {eval_type}')
                    if self.config.get('cross_domain'):
                        _, accuracies = self._validate_md(epoch_idx, eval_type)
                    else:
                        _, accuracies = self._validate(epoch_idx, eval_type)
                    test_accuracy, h = mean_confidence_interval(accuracies)
                    print("Test Accuracy: {:.3f}\t h: {:.3f}".format(test_accuracy, h))
                    total_accuracy += test_accuracy
                    total_accuracy_vector.extend(accuracies)
                    total_h[epoch_idx] = h

        aver_accuracy, h = mean_confidence_interval(total_accuracy_vector)
        print("Aver Accuracy: {:.3f}\t Aver h: {:.3f}".format(aver_accuracy, h))
        print("............Testing is end............")

        self._save_intermediate_logs()

        if self.writer is not None:
            self.writer.close()
            if self.distribute:
                dist.barrier()
        elif self.distribute:
            dist.barrier()

    def _validate(self, epoch_idx, eval_type):
        """
        The test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        # switch to evaluate mode
        self.model.eval()
        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        meter = self.test_meter
        meter.reset()
        episode_size = self.config["episode_size"]
        accuracies = []

        skip_iters = 0
        target_iters = 10 if (self.config['test_shot'] == 50) else 100
        print(target_iters)

        if (self.config.get('test_batch') != -1):
            skip_iters = self.config.get('test_batch')
        else:
            skip_iters = 0
        
        if (self.config['experiment_dir'] == 'bootstrapping_analysis') and (eval_type == 'bootstrapping'):
            for boot_round in range(1, 51):
                self.intermidiate_stats[eval_type][f"{boot_round}-round"] = []

        end = time()
        enable_grad = self.model_type != ModelType.METRIC
        log_scale = self.config["episode_size"]
        n_iters = 0
        with torch.set_grad_enabled(enable_grad):
            loader = self.test_loader
            for batch_idx, batch in enumerate(zip(*loader)):
                # skipping until the desired starting episode
                if batch_idx < skip_iters:
                    print(f"Iteration {batch_idx} skipped.")
                    continue
                
                if self.rank == 0:
                    self.writer.set_step(
                        int(
                            (
                                epoch_idx * len(self.test_loader)
                                + batch_idx * episode_size
                            )
                            * self.config["tb_scale"]
                        )
                    )

                meter.update("data_time", time() - end)

                # calculate the output
                if self.config.get('stability_test') == 'true':
                    batch = self._create_new_batch(batch)

                calc_begin = time()
                output_dict = self.model(
                    [elem for each_batch in batch for elem in each_batch]
                )

                first_key = list(output_dict)[0]
                self.intermidiate_stats[eval_type]['test_loss'].append(f"{output_dict[first_key]['loss'].detach():.3f}")
                self.intermidiate_stats[eval_type]['test_accuracy'].append(f"{output_dict[first_key]['accuracy']:.3f}")
                if (self.config['experiment_dir'] == 'bootstrapping_analysis') and (eval_type == 'bootstrapping'):
                    for key, value in output_dict[first_key]['episode_dict'].items():
                        self.intermidiate_stats[eval_type][f"{key}-round"].append(value)
                # print(self.intermidiate_stats[eval_type])

                accuracies.append(output_dict[first_key]['accuracy'])
                meter.update("calc_time", time() - calc_begin)

                # measure accuracy and record loss
                meter.update("acc", output_dict[first_key]['accuracy'])

                # measure elapsed time
                meter.update("batch_time", time() - end)

                if ((batch_idx + 1) * log_scale % self.config["log_interval"] == 0) or (
                    batch_idx + 1
                ) * episode_size >= max(map(len, loader)) * log_scale:
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Calc {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@1 {:.3f} ({:.3f})".format(
                            epoch_idx,
                            (batch_idx + 1) * log_scale,
                            max(map(len, loader)) * log_scale,
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("calc_time"),
                            meter.avg("calc_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc"),
                            meter.avg("acc"),
                        )
                    )
                    print(info_str)
                end = time()

                if (self.config.get('test_batch') != -1):
                    n_iters += 1
                    print(f"{n_iters}/{target_iters} iterations completed!")
                    if n_iters == target_iters:
                        break


        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        return meter.avg("acc"), accuracies

    def _validate_md(self, epoch_idx, eval_type):
        """
        The val/test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        # switch to evaluate mode
        self.model.eval()
        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        meter = self.test_meter
        meter.reset()
        episode_size = self.config["episode_size"]
        accuracies = []

        if (self.config.get('test_batch') != -1):
            n_iters = 50 if (self.config['test_shot'] == 50) else 100
            skip_iters = self.config.get('test_batch')
        else:
            n_iters = self.config['test_episode']
            skip_iters = 0

        print(n_iters)
        end = time()
        enable_grad = self.model_type != ModelType.METRIC
        log_scale = self.config["episode_size"]
        with torch.set_grad_enabled(enable_grad):
            loader = self.test_loader
            for batch_idx in range(n_iters):
                if ((skip_iters != 0) and (batch_idx == 0)):
                    for i in range(skip_iters):
                        print(f"Iteration {i} skipped.")
                        batch = next(loader)

                batch = next(loader)
                if self.rank == 0:
                    self.writer.set_step(
                        int(
                            (
                                epoch_idx * self.config['test_episode']
                                + batch_idx * episode_size
                            )
                            * self.config["tb_scale"]
                        )
                    )

                meter.update("data_time", time() - end)

                # calculate the output
                calc_begin = time()
                # if is_test:
                #     print([elem for each_batch in batch for elem in each_batch][1])
                # print(batch)
                output_dict = self.model(batch)
                # print(f"Epoch {epoch_idx} batch {batch_idx}")
                first_key = list(output_dict)[0]
                self.intermidiate_stats[eval_type]['test_loss'].append(f"{output_dict[first_key]['loss'].detach():.3f}")
                self.intermidiate_stats[eval_type]['test_accuracy'].append(f"{output_dict[first_key]['accuracy']:.3f}")
                accuracies.append(output_dict[first_key]['accuracy'])
                meter.update("calc_time", time() - calc_begin)

                # measure accuracy and record loss
                # print(first_key)
                # print(output_dict[first_key]['accuracy'])
                meter.update("acc", output_dict[first_key]['accuracy'])

                # measure elapsed time
                meter.update("batch_time", time() - end)

                if (((batch_idx + 1) % 100) == 0):
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Calc {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@1 {:.3f} ({:.3f})\t".format(
                            epoch_idx,
                            (batch_idx + 1),
                            self.config['test_episode'],
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("calc_time"),
                            meter.avg("calc_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc"),
                            meter.avg("acc"),
                        )
                    )
                    print(info_str)
                end = time()

                if (eval_type == 'cross_val-1000fold') or (eval_type == 'bootstrapping'):
                    print(f"Iterations done: {batch_idx}.")

        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        return meter.avg("acc"), accuracies

    def _init_files(self, config):
        """
        Init result_path(log_path, viz_path) from the config dict.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (viz_path, checkpoints_path).
        """
        if self.result_path is not None:
            result_path = self.result_path
        else:
            result_dir = "{}-{}-{}-{}-{}".format(
                config["classifier"]["name"],
                # you should ensure that data_root name contains its true name
                config["data_root"].split("/")[-1],
                config["backbone"]["name"],
                config["way_num"],
                config["shot_num"],
            )
            result_path = os.path.join(config["result_root"], result_dir)
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")

        init_logger_config(
            config["log_level"],
            log_path,
            config["classifier"]["name"],
            config["backbone"]["name"],
            is_train=False,
            rank=self.rank,
        )

        state_dict_path = os.path.join(result_path, "checkpoints", "model_best.pth")
        if self.rank == 0:
            create_dirs([result_path, log_path, viz_path])

        return viz_path, state_dict_path

    def _init_logger(self):
        self.logger = getLogger(__name__)

        # Hack print
        def use_logger(msg, level="info"):
            if self.rank != 0:
                return
            if level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            else:
                raise ("Not implemente {} level log".format(level))

        builtins.print = use_logger

        return self.logger

    def _init_dataloader(self, config, cross_domain=False):
        """
        Init dataloaders.(train_loader, val_loader and test_loader)

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (train_loader, val_loader and test_loader).
        """
        if cross_domain == None:
            cross_domain = False
        self._check_data_config()
        
        if cross_domain:
            episodes_config_test = {}
            episodes_config_test["n_way"] = self.config['test_way']
            episodes_config_test["min_ways"] = self.config.get('min_way_test') if self.config.get('min_way_test') else self.config['way_num']
            episodes_config_test["max_ways"] = self.config.get('max_way_test') if self.config.get('max_way_test') else self.config['way_num']
            episodes_config_test["k_shot"] = self.config['test_shot']
            episodes_config_test["min_shots"] = self.config.get('min_shot_test') if self.config.get('min_shot_test') else self.config['shot_num']
            episodes_config_test["max_shots"] = self.config.get('max_shot_test') if self.config.get('max_shot_test') else self.config['shot_num']
            episodes_config_test["query_size"] = self.config['test_query']
    
            print(self.config['test_datasets'])
            # test_loader = iter(DataLoaderCrossDomain(create_datasets(self.config['test_datasets'], self.config['data_root']), self.config['test_episode'], episodes_config_test, test_loader=True).generator(self.config['seed']))
            print(f"Tasks per dataset: {int(self.config['test_episode'] / 10)}")
            test_loader = iter(DataLoaderCrossDomain(create_datasets(self.config['test_datasets'], self.config['data_root']), int(self.config['test_episode'] / 10), episodes_config_test, test_loader=True).generator(self.config['seed']))
        else:
            distribute = self.distribute
            test_loader = get_dataloader(config, "test", self.model_type, distribute)

        return test_loader

    def _check_data_config(self):
        """
        Check the config params.
        """
        # check: episode_size >= n_gpu and episode_size != 0
        assert (
            self.config["episode_size"] >= self.config["n_gpu"]
            and self.config["episode_size"] != 0
        ), "episode_size {} should be >= n_gpu {} and != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_size % n_gpu == 0
        assert (
            self.config["episode_size"] % self.config["n_gpu"] == 0
        ), "episode_size {} % n_gpu {} != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_num % episode_size == 0
        assert (
            self.config["train_episode"] % self.config["episode_size"] == 0
        ), "train_episode {} % episode_size  {} != 0".format(
            self.config["train_episode"], self.config["episode_size"]
        )

        assert (
            self.config["test_episode"] % self.config["episode_size"] == 0
        ), "test_episode {} % episode_size  {} != 0".format(
            self.config["test_episode"], self.config["episode_size"]
        )

    def _init_model(self, config):
        """
        Init model (backbone+classifier) from the config dict and load the best checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        emb_func = get_instance(arch, "backbone", config)
        model_kwargs = {
            "way_num": config["way_num"],
            "shot_num": config["shot_num"] * config["augment_times"],
            "query_num": config["query_num"],
            "test_way": config["test_way"],
            "test_shot": config["test_shot"] * config["augment_times"],
            "test_query": config["test_query"],
            "eval_types" : config["eval_types"],
            "k_fold" : config["k_fold"],
            "boot_rounds" : config["boot_rounds"],
            "boot_seed" : config.get("boot_seed"),
            "emb_func": emb_func,
            "device": self.device,
        }
        model = get_instance(arch, "classifier", config, **model_kwargs)

        print(model)
        print("Trainable params in the model: {}.".format(count_parameters(model)))
        print("Loading the state dict from {}.".format(self.state_dict_path))
        state_dict = torch.load(self.state_dict_path, map_location="cpu")
        # print(state_dict)
        if config['classifier']['name'] == 'MAML' and (config.get('fixed_classes') and (config.get('fixed_classes') != 0)):
            layers_to_remove = ['classifier.layers.0.weight', 'classifier.layers.0.bias']
            for key in layers_to_remove:
                del state_dict[key]
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)


        if self.distribute:
            # higher order grad of BN in multi gpu will conflict with syncBN
            # FIXME MAML with multi GPU is conflict with syncBN
            if not (
                self.config["classifier"]["name"] in ["MAML"]
                and self.config["n_gpu"] > 1
            ):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                print(
                    "{} with multi GPU will conflict with syncBN".format(
                        self.config["classifier"]["name"]
                    ),
                    level="warning",
                )
            model = model.to(self.rank)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True,
            )

            return model, model.module.model_type
        else:
            model = model.to(self.device)

            return model, model.model_type

    def _init_device(self, rank, config):
        """
        Init the devices from the config file.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of devices and list_ids.
        """
        init_seed(config["seed"], config["deterministic"])
        device, list_ids = prepare_device(
            rank,
            config["device_ids"],
            config["n_gpu"],
            backend="nccl"
            if "dist_backend" not in self.config
            else self.config["dist_backend"],
            dist_url="tcp://127.0.0.1:" + str(config["port"])
            if "dist_url" not in self.config
            else self.config["dist_url"],
        )
        torch.cuda.set_device(self.rank)

        return device, list_ids

    def _init_meter(self):
        """
        Init the AverageMeter of test stage to cal avg... of batch_time, data_time, calc_time and acc.

        Returns:
            AverageMeter: Test_meter.
        """
        test_meter = AverageMeter(
            "test", ["batch_time", "data_time", "calc_time", "acc"], self.writer
        )

        return test_meter

    def _init_writer(self, viz_path):
        """
        Init the tensorboard writer.

        Return:
            writer: tensorboard writer
        """
        if self.rank == 0:
            writer = TensorboardWriter(viz_path)
            return writer
        else:
            return None
    
    def _create_new_batch(self, batch):
        total_samples = self.config['test_shot'] + self.config['test_query'] + 1
        swap_class = np.random.randint(self.config['test_way'], size=1)
        swap_element = np.random.randint(self.config['test_shot'], size=1)
        swap_element_idx = int((swap_class * (self.config['test_shot'] + self.config['test_query'])) + swap_element)
        new_element_idx = int(((swap_class + 1) * total_samples) - 1)
        # print(swap_element_idx)
        # print(new_element_idx)

        batch_labels = [elem for each_batch in batch for elem in each_batch][1][0]
        batch_images = [elem for each_batch in batch for elem in each_batch][0]
        batch_images_idx = torch.tensor(list((range(self.config['test_way'] * (self.config['test_shot'] + self.config['test_query'] + 1)))))
        batch_images_idx[swap_element_idx] = new_element_idx
        batch_images_idx[new_element_idx] = swap_element_idx
        batch_images = batch_images[batch_images_idx]
        images_exclude_idx = torch.tensor([20, 41, 62, 83, 104])

        mask = torch.ones_like(batch_images_idx).scatter_(0, images_exclude_idx, 0.)
        mask = mask[:, None, None, None]
        mask = mask.expand(batch_images.size())
        batch_images = torch.masked_select(batch_images, (mask == 1)).view((self.config['test_shot'] + self.config['test_query']) * self.config['test_way'], 3, 84, 84)
        batch_labels = batch_labels[:, :-1]
        new_batch = ([batch_images, torch.unsqueeze(batch_labels, 0)],)

        return new_batch

    def _init_result_root(self):
        if "/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE" in self.config['result_root']:
            self.config['result_root'] = self.config['result_root'].replace("/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE", "/mnt/invinciblefs/scratch/lushima" )
        elif "/home/s2589574/BEPE" in self.config['result_root']:
            self.config['result_root'] = self.config['result_root'].replace("/home/s2589574/BEPE", "/mnt/invinciblefs/scratch/lushima" )

    def _create_intermidiate_dirs(self):
        # Creating model directory if it doesn't already exist
        if not os.path.isdir(self.config["result_root"]):
            os.mkdir(self.config["result_root"])

        # Creating dataset directory if it doesn't already exist
        result_path = os.path.join(self.config["result_root"], self.config["data_root"].split("/")[-1])
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        # Creating n-way k-shot directory
        result_path = os.path.join(result_path, f"{self.config['test_way']}way-{self.config['test_shot']}shot")
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        
        # if experimental setup
        if ((self.config.get('fixed_classes') and (self.config.get('fixed_classes') != 0)) or (self.config.get('stability_test') == 'true') or (self.config['experiment_dir'] == 'bootstrapping_analysis')):
            result_path = os.path.join(result_path, self.config['experiment_dir'])
            if not os.path.isdir(result_path):
                os.mkdir(result_path)
        
        return result_path

    def _save_experiment_metadata(self, save_path):
        experiment_dict = {}
        experiment_dict['config_path'] = self.result_path
        experiment_dict['way_num'] = self.config['way_num']
        experiment_dict['shot_num'] =  self.config['shot_num']
        experiment_dict['test_way'] =  self.config['test_way']
        experiment_dict['test_shot'] =  self.config['test_shot']
        experiment_dict['dataset'] =  self.config['data_root']
        experiment_dict['date_time'] = get_local_time()

        with open(os.path.join(save_path, 'metadata.yaml'), 'w') as metadata_file:
            print(os.path.join(save_path, 'metadata.yaml'))
            yaml.dump(experiment_dict, metadata_file)
            print(f"Metadata file saved successfully!")
    
    def _save_intermediate_logs(self):
        result_path = self._create_intermidiate_dirs()
        self._save_experiment_metadata(result_path)

        for eval_type in self.intermidiate_stats.keys():
            threshold = len(self.intermidiate_stats[eval_type]['test_loss'])

            # self.intermidiate_stats[eval_type]['train_loss'] = self.intermidiate_stats[eval_type]['train_loss'][:threshold]
            # self.intermidiate_stats[eval_type]['train_accuracy'] = self.intermidiate_stats[eval_type]['train_accuracy'][:threshold]
            data = pd.DataFrame.from_dict(self.intermidiate_stats[eval_type])
            # if eval_type == 'cross_val':
                # data.to_csv(os.path.join(result_path, f'{self.config["classifier"]["name"]}-{self.config["k_fold"]}fold-{eval_type}.csv'), sep=',', index=False, header=True)
            # else:
            if (self.config.get('fixed_classes') and (self.config.get('fixed_classes') != 0)):
                data.to_csv(os.path.join(result_path, f"{self.config['classifier']['name']}-{self.config['test_way']}way_{self.config['test_shot']}shot-{eval_type}.csv"), sep=',', index=False, header=True)
            elif self.config.get('cross_domain'):
                if (self.config.get('test_batch') and (self.config.get('test_batch') != -1)):
                    data.to_csv(os.path.join(result_path, f"{self.config['classifier']['name']}-{eval_type}-{self.config['val_datasets'][0][-8:]}-{self.config.get('test_batch')}.csv"), sep=',', index=False, header=True)
                else:
                    data.to_csv(os.path.join(result_path, f"{self.config['classifier']['name']}-{eval_type}-{self.config['val_datasets'][0][-8:]}.csv"), sep=',', index=False, header=True)
            elif (self.config.get('test_batch') and (self.config.get('test_batch') != -1)):
                data.to_csv(os.path.join(result_path, f"{self.config['classifier']['name']}-{eval_type}-{self.config.get('test_batch')}.csv"), sep=',', index=False, header=True)
            elif (self.config['experiment_dir'] == 'bootstrapping_analysis'):
                data.to_csv(os.path.join(result_path, f"{self.config['classifier']['name']}-{eval_type}-{self.config.get('boot_seed')}seed.csv"), sep=',', index=False, header=True)
            else:
                data.to_csv(os.path.join(result_path, f'{self.config["classifier"]["name"]}-{eval_type}.csv'), sep=',', index=False, header=True)