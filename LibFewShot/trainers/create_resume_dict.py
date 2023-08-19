import yaml
import argparse
import os

# PATH = "/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/MAML/miniImageNet/5way-5shot/MAML-miniImageNet-Conv64F-5-5-Apr-20-2023-11-10-32"
# VAR_DICT = {
#     "device_ids": "0,2",
#     "n_gpu": 2,
#     "test_episode": 600,
#     "episode_size": 2,
#     "eval_types": 'oracle, bootstrapping',
#     "k_fold": 5
# }

# PATH = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models'
PATH = '/mnt/invinciblefs/scratch/lushima/Models'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--way", type=int)
    parser.add_argument("--shot", type=int)
    args = parser.parse_args()

    resume_dict = {}
    config_path = f'{PATH}/{args.model}/{args.dataset}/{args.way}way-{args.shot}shot/{args.path}'
    resume_dict['path'] = config_path

    with open(f"./trainers/aux.yaml", 'w') as file:
        yaml.dump(resume_dict, file)

    print(resume_dict)

if __name__ == '__main__':
    main()