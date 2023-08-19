import torch 
import numpy as np

import torch
import torchvision.transforms as T
from PIL import Image

from utils import create_support_prompt, create_query_prompt, trunc_caption, display_interleaved_outputs, create_split_prompts

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def oracle_eval(self, support_set, support_target, query_set, query_target):
        support_prompt = create_support_prompt(support_set, support_target, self.model.dataset_idx_to_label)
        accuracy = 0

        n_samples = 0
        for query_image, query_label in zip(query_set, query_target):
            query_prompt = create_query_prompt(support_prompt, query_image)
            # print('Prompt:')
            # display_interleaved_outputs(query_prompt)
            # print('=' * 30)
            model_outputs = self.model.model.generate_for_images_and_texts(query_prompt, num_words=32, ret_scale_factor=0, max_img_per_ret=3)
            print('Model generated outputs:')
            print(model_outputs)
            if self.model.dataset_idx_to_label[int(query_label)].lower() in model_outputs[0].lower():
                print("Acertou!")
                accuracy += 100
            print(self.model.dataset_idx_to_label[int(query_label)].lower())   
            n_samples += 1
            print(n_samples)      

        mean_accuracy = accuracy / (self.model.config['test_query'] * self.model.config['test_way'])   
        print(f"Final oracle accuracy: {mean_accuracy:.2f}%")
        return mean_accuracy

    def cv_eval(self, support_set, support_target, query_set, query_target, k_fold=6):
        support_prompt = create_support_prompt(support_set, support_target)
        accuracy = 0

        elems_per_split, splitted_support_promps = create_split_prompts(support_prompt, self.model.config['test_way'] * self.model.config['test_shot'], k_fold)
        for fold in range(k_fold):
            query_cv = splitted_support_promps[fold]
            support_prompt_cv = splitted_support_promps[:fold] + splitted_support_promps[fold+1:]
            for elem in range(0, elems_per_split[fold], 2):
                query_prompt = create_query_prompt(support_prompt_cv, query_cv[elem])
                display_interleaved_outputs(query_prompt)
                print('=' * 30)
                model_outputs = self.model.generate_for_images_and_texts(query_prompt, num_words=32, ret_scale_factor=0, max_img_per_ret=3)
                print('Model generated outputs:')
                print(model_outputs)
                if self.model.dataset_idx_to_label[query_label].lower() in model_outputs.lower():
                    print("Acertou!")
                    accuracy += 100


        for query_image, query_label in zip(query_set, query_target):
            query_prompt = create_query_prompt(support_prompt, query_image)
            print('Prompt:')
            
        print(f"Final oracle accuracy: {accuracy / self.model.config['test_query']:.2f}%")
        return accuracy