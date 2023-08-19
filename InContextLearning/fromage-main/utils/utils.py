import numpy as np
import copy
import torch
from transformers import logging
logging.set_verbosity_error()

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import pickle

def convert_to_PIL_image(image):
    t = T.ToPILImage()
    return t(image)

def split_support_query(images, labels, config):
    support_idx = torch.arange(config['test_shot'])
    aux = torch.arange(config['test_shot'])
    for _ in range(config['test_way']-1):
        aux += config['test_query'] + config['test_shot']
        support_idx = torch.cat([support_idx, aux])
    support_mask = torch.zeros(labels.size(), dtype=torch.bool)
    support_mask[support_idx] = True
    query_mask = support_mask == False
    support_set, support_target, query_set, query_target = images[support_mask], labels[support_mask], images[query_mask], labels[query_mask]
    return support_set, support_target, query_set, query_target

def get_shuffle(n_way, k_shot):
    shuffle_idx = torch.arange(n_way) * k_shot
    aux = torch.arange(n_way) * k_shot
    for _ in range(k_shot-1):
        aux += 1
        shuffle_idx = torch.cat([shuffle_idx, aux])
    
    return shuffle_idx

def split_data(batch, config):
    images, labels = batch
    support_set, support_target, query_set, query_target = split_support_query(images, labels, config)

    intercalated_idx = get_shuffle(config['test_way'], config['test_shot'])

    return support_set[intercalated_idx], support_target[intercalated_idx], query_set, query_target

def create_support_prompt(support_set, support_target, idx_to_label):
    prompt = []
    for image, label in zip(support_set, support_target):
        prompt.append(convert_to_PIL_image(image))
        prompt.append(f"This is a {idx_to_label[int(label)]}.")
    return prompt

def create_query_prompt(support_prompt, query_image):
    query_prompt = support_prompt.copy()
    query_prompt.append(convert_to_PIL_image(query_image))
    query_prompt.append('Question: What is this? \nAnswer:')

    return query_prompt

def create_split_prompts(support_set, n_elements, n_chunks):
    split_set = []
    elements_per_chunk = n_elements // n_chunks
    split_list = [elements_per_chunk] * n_chunks
    remainder = n_elements - (elements_per_chunk * n_chunks)

    for idx in range(remainder):
        split_list[idx] += 1
    
    beg = 0
    for n_elem in split_list:
        split_set.append([support_set[beg:beg+(n_elem*2)]])
        beg += n_elem*2

    return split_list, split_set

def trunc_caption(caption: str) -> str:
    # Truncate at period.
    trunc_index = caption.find('.') + 1
    if trunc_index < 0:
        trunc_index = caption.find('\n') + 1
    caption = caption[:trunc_index]
    return caption

def display_interleaved_outputs(model_outputs, one_img_per_ret=True):
    for output in model_outputs:
        if type(output) == str:
            print(output)
        elif type(output) == list:
            if one_img_per_ret:
                plt.figure(figsize=(3, 3))
                plt.imshow(np.array(output[0]))
            else:
                fig, ax = plt.subplots(1, len(output), figsize=(3 * len(output), 3))
                for i, image in enumerate(output):
                    image = np.array(image)
                    ax[i].imshow(image)
                    ax[i].set_title(f'Retrieval #{i+1}')
            plt.show()
        elif type(output) == Image.Image:
            plt.figure(figsize=(3, 3))
            plt.imshow(np.array(output))
            plt.show()


