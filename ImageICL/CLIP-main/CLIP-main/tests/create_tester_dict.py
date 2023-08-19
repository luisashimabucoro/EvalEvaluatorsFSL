import yaml
import argparse
import os

def get_cur_path():
    """Get the absolute path of current file.

    Returns: The absolute path of this file (Config.py).

    """
    return os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_epoch", type=int, default=1)
    parser.add_argument("--test_episode", type=int, default=600)
    parser.add_argument("--experiment_dir", type=str)
    parser.add_argument("--fixed_classes", type=int)
    parser.add_argument("--stability_test", type=str)
    parser.add_argument("--test_batch", type=int)

    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--test_way", type=int)
    parser.add_argument("--way_num", type=int)
    parser.add_argument("--test_shot", type=int)
    parser.add_argument("--shot_num", type=int)
    parser.add_argument("--boot_seed", type=int)

    parser.add_argument("--n_gpu", type=int)
    parser.add_argument("--device_ids", type=str)
    parser.add_argument("--eval_types", type=str)
    parser.add_argument("--test_folds", type=str)
    args = parser.parse_args()

    config_path = f'{get_cur_path}/../reproduce'
    print(os.listdir(config_path))
    print(sorted(os.listdir(config_path)))

    dict_test = vars(args)
    # dict_test['config_path'] = os.path.join(config_path, sorted(os.listdir(config_path))[0])
    # dict_test['config_path'] = os.path.join(config_path, sorted(os.listdir(config_path))[1])
    dict_test['config_path'] = os.path.join(config_path, sorted(os.listdir(config_path))[-1])

    with open(f"./testers/{dict_test['model']}-{dict_test['dataset']}.yaml", 'w') as file:
        del dict_test['model']
        del dict_test['dataset']
        yaml.dump(dict_test, file)
    print(dict_test)

if __name__ == '__main__':
    main()
