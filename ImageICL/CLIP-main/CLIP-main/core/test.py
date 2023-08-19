import numpy as np
import torch
from pkg_resources import packaging
import clip
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from collections import OrderedDict
import torch
import zipfile
import pickle
from nltk.corpus import wordnet as wn
from core.utils import split_data, Evaluator, get_local_time
from time import time

import yaml
from core.dataloader import DataLoaderCrossDomain, create_datasets

class InContextModel(object):
    def __init__(self, config):
        self.config = config
        self.model, self.preprocess = clip.load(self.config['backbone'])
        self.model.cuda().eval()
        self.evaluator = Evaluator(self)

def create_dataloader(config):
        episodes_config_test = {}
        episodes_config_test["n_way"] = config['test_way']
        episodes_config_test["min_ways"] = config.get('min_way_test') if config.get('min_way_test') else config['test_way']
        episodes_config_test["max_ways"] = config.get('max_way_test') if config.get('max_way_test') else config['test_way']
        episodes_config_test["k_shot"] = config['test_shot']
        episodes_config_test["min_shots"] = config.get('min_shot_test') if config.get('min_shot_test') else config['test_shot']
        episodes_config_test["max_shots"] = config.get('max_shot_test') if config.get('max_shot_test') else config['test_shot']
        episodes_config_test["query_size"] = config['test_query']

        # test_loader = iter(DataLoaderCrossDomain(create_datasets(config['test_datasets'], config['data_root']), config['test_episode'], episodes_config_test, test_loader=True).generator(config['seed']))
        test_loader = iter(DataLoaderCrossDomain(create_datasets(config['test_datasets'], config['data_root']), int(config['test_episode']), episodes_config_test, test_loader=False).generator(42))
        
        with open(f"/home/s2589574/BEPE/EmbeddingFSL/CLIP-main/CLIP-main/data/{config['test_datasets'][0]}.pickle", 'rb') as handle:
            idx_to_label = pickle.load(handle)

        return test_loader, idx_to_label

class Tester(object):
     
    def __init__(self, config):
        self.config = config
        self.dataloader, self.idx_to_label = create_dataloader(config)
        self.model = InContextModel(config)
        self.result_path = os.path.join(self.config['result_root'], self.config['backbone'].replace('/', ''))
        self.intermidiate_stats = {}

    def _save_experiment_metadata(self, save_path):
        experiment_dict = {}
        experiment_dict['config_path'] = self.result_path
        experiment_dict['test_way'] = self.config['test_way']
        experiment_dict['test_shot'] =  self.config['test_shot']
        experiment_dict['dataset'] =  os.path.join(self.config['data_root'], self.config['test_datasets'][0])
        experiment_dict['date_time'] = get_local_time()

        with open(os.path.join(save_path, 'metadata.yaml'), 'w') as metadata_file:
            print(os.path.join(save_path, 'metadata.yaml'))
            yaml.dump(experiment_dict, metadata_file)
            print(f"Metadata file saved successfully!")

    def _create_intermidiate_dirs(self):
        # Creating model directory if it doesn't already exist
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

        # Creating dataset directory if it doesn't already exist
        result_path = os.path.join(self.result_path, self.config["test_datasets"][0])
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        # Creating n-way k-shot directory
        result_path = os.path.join(result_path, f"{self.config['test_way']}way-{self.config['test_shot']}shot")
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        
        return result_path
    
    def _save_intermediate_logs(self):
        result_path = self._create_intermidiate_dirs()
        self._save_experiment_metadata(result_path)

        for eval_type in self.intermidiate_stats.keys():

            data = pd.DataFrame.from_dict(self.intermidiate_stats[eval_type])
            data.to_csv(os.path.join(result_path, f'CLIP-{eval_type}.csv'), sep=',', index=False, header=True)
    
    def test_loop(self):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """

        for eval_type in self.config['eval_types']:
            config = self.config.copy()
            config['eval_types'] = [eval_type]
            
            if eval_type == 'cross_val':
                for fold in self.config['test_folds']:
                    self.dataloader, _ = create_dataloader(config)
                    # setting loo cross validation value before testing
                    fold_value = (config['test_way'] * config['test_shot']) if (int(fold) == 1000) else int(fold)
                    config['k_fold'] = fold_value
                    self.intermidiate_stats[f'{eval_type}-{fold}fold'] = {'test_accuracy' : []}
                    print("============ Testing on the test set ============")
                    print(f"Evaluation: {eval_type} with {config['k_fold']} folds") 
                    avg_acc = self._validate(config, f'{eval_type}-{fold}fold')
                     
            else:
                self.dataloader, _ = create_dataloader(self.config)
                self.intermidiate_stats[f'{eval_type}'] = {'test_accuracy' : []}
                print("============ Testing on the test set ============")
                print(f'Evaluation: {eval_type}')
                avg_acc = self._validate(config, eval_type)
                  

        self._save_intermediate_logs()
        print("............Testing is end............")


    def _validate(self, config, eval_type):

        start_time = time()
        for episode_idx in range(config['test_episode']):
            batch = next(self.dataloader)
            support_set, support_target, query_set, query_target = split_data(batch, config)
            output_dict = self.model.evaluator.eval_batch(config, support_set, support_target, query_set, query_target, config.get('k_fold'), config.get('boot_rounds'))
            
            first_key = list(output_dict)[0]
            self.intermidiate_stats[eval_type]['test_accuracy'].append(output_dict[first_key]['accuracy'])
            if ((len(self.intermidiate_stats[eval_type]['test_accuracy']) != 0) and (len(self.intermidiate_stats[eval_type]['test_accuracy']) % 100 == 0)):
                end_time = time()
                print(f"{len(self.intermidiate_stats[eval_type]['test_accuracy'])}/{config['test_episode']} - Time: {(end_time-start_time)/60:.2f}min | Avg Acc: {sum(self.intermidiate_stats[eval_type]['test_accuracy']) / len(self.intermidiate_stats[eval_type]['test_accuracy']):.2f}")
                start_time = end_time

        return sum(self.intermidiate_stats[eval_type]['test_accuracy']) / len(self.intermidiate_stats[eval_type]['test_accuracy'])

def main():
    print("Torch version:", torch.__version__)
    print(clip.available_models())
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    dataloader = create_dataloader(config)

    # print(wn.synset_from_pos_and_offset('n',4543158))
    clip_model = InContextModel(config)

    input_resolution = clip_model.model.visual.input_resolution
    context_length = clip_model.model.context_length
    vocab_size = clip_model.model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    result_list = []
    for batch_idx in range(config['test_episode']):
        start_time = time()
        batch = next(dataloader)
        support_set, support_target, query_set, query_target = split_data(batch, config)
        eval_results = clip_model.evaluator.eval_batch(support_set, support_target, query_set, query_target, k_fold, boot_rounds=30)
        oracle_result_dict = clip_model.evaluator.oracle_evaluation(support_set, support_target, query_set, query_target)
        boot_result_dict = clip_model.evaluator.bootstrapping_evaluation(support_set, support_target, boot_rounds=30)
        print(f"Bootstrapping accuracy: {boot_result_dict['accuracy']}")
        print(f"MAE {abs(oracle_result_dict['accuracy'] - boot_result_dict['accuracy'])}")


# if __name__ == "__main__":
#     main()