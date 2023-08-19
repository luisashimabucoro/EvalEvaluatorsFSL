# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Tester
import os 

def get_cur_path():
    """Get the absolute path of current file.

    Returns: The absolute path of this file (Config.py).

    """
    return os.path.dirname(__file__)


def main(config):
    tester = Tester(config)
    tester.test_loop()

if __name__ == "__main__":
    config = Config(os.path.join(get_cur_path(), 'reproduce/config.yaml')).get_config_dict()
    main(config)