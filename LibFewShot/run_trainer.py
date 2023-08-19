# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer
from core import Test
import argparse

import pandas as pd


def main(rank, config):
    trainer = Trainer(rank, config)
    result_path, storage_path = trainer.train_loop(rank)

    if config.get('allow_test'):
        config = Config(os.path.join(result_path, 'config.yaml')).get_config_dict()
        test = Test(rank, config, storage_path)
        test.test_loop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", "--config_path", type=str)
    parser.add_argument("-resume", "--resume", action='store_true', help='flag to resume training')
    parser.add_argument("-allow_test", "--allow_test", action='store_true', help='flag indicating performance evaluation \
                                                                        should be done after the model is done training')
    args = parser.parse_args()

    config = Config(args.config_path, is_resume=args.resume).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)