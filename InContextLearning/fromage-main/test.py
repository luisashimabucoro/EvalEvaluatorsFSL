import numpy as np
import copy
import torch
import transformers
# logging.set_verbosity_error()

from PIL import Image
import matplotlib.pyplot as plt
import pickle

from fromage import models
from fromage import utils

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import yaml
from dataloader import DataLoaderCrossDomain, create_datasets, MetaAlbumDataset
from utils import split_data, Evaluator, create_support_prompt, create_query_prompt, trunc_caption, display_interleaved_outputs, create_split_prompts

class InContextModel(object):
    def __init__(self, model_dir, config, idx_to_label):
        self.model = models.load_fromage(model_dir).to(dtype=torch.float16)
        self.config = config
        self.dataset_idx_to_label = idx_to_label
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

        print(config['test_datasets'])
        # test_loader = iter(DataLoaderCrossDomain(create_datasets(config['test_datasets'], config['data_root']), config['test_episode'], episodes_config_test, test_loader=True).generator(config['seed']))
        print(f"Tasks per dataset: {int(config['test_episode'])}")
        test_loader = iter(DataLoaderCrossDomain(create_datasets(config['test_datasets'], config['data_root']), int(config['test_episode']), episodes_config_test, test_loader=True).generator(42))
        return test_loader

def main():
    # Load model used in the paper.
    model_dir = './fromage_model/fromage_vis4'
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    dataloader = create_dataloader(config)
    with open(f"/home/s2589574/BEPE/InContextLearning/fromage-main/datasets/{config['test_datasets'][0]}.pickle", 'rb') as handle:
        idx_to_label = pickle.load(handle)
    model = InContextModel(model_dir, config, idx_to_label)
    # result_list = []
    # for batch_idx in range(config['test_episode']):
    #     prompt = []
    #     batch = next(dataloader)
    #     support_set, support_target, query_set, query_target = split_data(batch, config)
    #     acc = model.evaluator.oracle_eval(support_set, support_target, query_set, query_target)
    #     result_list.append(acc)

    #     # display_interleaved_outputs(prompt)
    #     # print('=' * 30)
    #     # model_outputs = model.generate_for_images_and_texts(prompt, num_words=32, ret_scale_factor=0, max_img_per_ret=3)

    #     # Display outputs.
    #     # print('Model generated outputs:')
    #     # print(model_outputs)
    # print(result_list)
    # print(f"Mean accuracy: {sum(result_list) / len(result_list)}")
    # return
    # Load an image of a cat.
    # inp_image = utils.get_image_from_url('https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg')
    cat1 = utils.get_image_from_url('https://i2-prod.gloucestershirelive.co.uk/news/gloucester-news/article6728882.ece/ALTERNATES/s1200c/0_WMPS-West-Midland-Safari-Parkjp.jpg')
    dog1 = utils.get_image_from_url('https://www.akc.org/wp-content/uploads/2017/11/Siberian-Husky-standing-outdoors-in-the-winter.jpg')
    cat2 = utils.get_image_from_url('https://www.colchester-zoo.com/wp-content/uploads/2022/03/Bailey-Statement.jpg')
    dog2 = utils.get_image_from_url('https://www.wideopenpets.com/wp-content/uploads/sites/6/2019/08/Owning-a-Husky.png')
    dog_question = utils.get_image_from_url('https://www.insidedogsworld.com/wp-content/uploads/2016/12/husky10-564x600.jpg')
    task_induction = 'Answer with dog or lion.'
    cat_prompt = 'This is a lion.'
    dog_prompt = 'This is a dog.'
    question = 'Q: What is this? \nA: this is a '
    # Get FROMAGe to retrieve images of cats in other styles.
    # for inp_text in ['watercolor drawing [RET]']:
    #     prompt = [inp_image, inp_text]
    #     print('Prompt:')
    #     display_interleaved_outputs(prompt)
    #     print('=' * 30)
    #     model_outputs = model.generate_for_images_and_texts(prompt, max_img_per_ret=3)

    #     # Display outputs.
    #     print('Model generated outputs:')
    #     display_interleaved_outputs(model_outputs, one_img_per_ret=False)

    prompt = [task_induction, cat1, cat_prompt, dog1, dog_prompt, cat2, cat_prompt, dog2, dog_prompt, dog_question, question]
    print('Prompt:')
    display_interleaved_outputs(prompt)
    print('=' * 30)
    model_outputs = model.model.generate_for_images_and_texts(prompt, num_words=32, ret_scale_factor=0, max_img_per_ret=3)

    # Display outputs.
    print('Model generated outputs:')
    print(model_outputs)
    # display_interleaved_outputs(model_outputs, one_img_per_ret=False)

if __name__ == '__main__':
    main()