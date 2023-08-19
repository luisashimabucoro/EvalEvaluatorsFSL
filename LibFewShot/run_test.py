import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Test
import argparse

import pandas as pd


def main(rank, config, result_path):
    test = Test(rank, config, result_path)
    test.test_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", "--config_path", help='define the path where the config file is', type=str)
    parser.add_argument("-resume", "--resume", action='store_true', help='flag to resume training')
    args = parser.parse_args()

    config = Config(args.config_path).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config, os.path.dirname(args.config_path)))
    else:
        main(0, config, os.path.dirname(args.config_path))