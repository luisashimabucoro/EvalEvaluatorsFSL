import numpy as np
import copy
import torch
from transformers import logging
from datetime import datetime
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

    return support_set, support_target, query_set, query_target

def create_labels(labels, n_way):
    unique_labels = torch.unique(labels)
    unique_fabricated_labels = torch.arange(n_way)

    for og_label, new_label in zip(unique_labels, unique_fabricated_labels):
        labels[labels == og_label] = new_label
    
    return labels

def get_img_emb(support_set, clip_model):
    with torch.no_grad():
        img_list = []
        for img in support_set:
            img = convert_to_PIL_image(img)
            img_list.append(clip_model.preprocess(img))

        img_list = torch.stack(img_list).cuda()    
        image_emb = clip_model.model.encode_image(img_list).float()
    
        return image_emb

def create_split_list(n_elements, n_chunks):
    split_set = []
    elements_per_chunk = n_elements // n_chunks
    split_list = [elements_per_chunk] * n_chunks
    remainder = n_elements - (elements_per_chunk * n_chunks)

    for idx in range(remainder):
        split_list[idx] += 1

    return split_list

def get_local_time():
    cur_time = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")

    return cur_time


