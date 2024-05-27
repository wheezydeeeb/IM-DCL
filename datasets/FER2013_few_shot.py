# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *

identity = lambda x:x

class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        d = ImageFolder(FER2013_path)

        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class SetDataset:
    def __init__(self, n_way, n_samples, transform):
        self.n_way = n_way
        self.n_samples = n_samples
        self.transform = transform

        d = ImageFolder(FER2013_path)
        self.data = []
        self.labels = []
        for i, (data, label) in enumerate(d):
            self.data.append(data)
            self.labels.append(label)

    def __getitem__(self, batch):
        x = []
        for way in batch:
            way_data = []
            for i in range(self.n_way):
                way_data.append(self.transform(self.data[self.labels.index(i)][way]))
            x.append(torch.stack(way_data))
        x = torch.stack(x)
        return x

    def __len__(self):
        return len(self.labels)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, n_shot, n_query):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.n_shot = n_shot
        self.n_query = n_query

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            batch = torch.randperm(self.n_classes)[:self.n_way]
            yield batch

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size), int(self.image_size)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
            # transform_list = ['ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.n_way, self.n_support + self.n_query, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide, self.n_support, self.n_query)  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 2, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    pass