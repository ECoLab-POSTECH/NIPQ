from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from .cutout import Cutout
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

transform_train = \
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
        Cutout(8)
    ])
    

transform_test = \
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
    ])
    
    
class SubsetSequantialSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in len(self.indices))

    def __len__(self):
        return len(self.indices)    


def get_loader(root, test_batch, train_batch, 
    valid_size=0, valid_batch=0, valid_shuffle=False, workers=4, seed=12345):
    
    # separate training set for validation if necessary
    indices = list(range(50000))    
    np.random.seed(seed)
    np.random.shuffle(indices)            
    train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
    
    loaders = {}
    
    # validation set
    if valid_size > 0:
        if valid_batch <= 0:
            valid_batch = test_batch
        
        if valid_shuffle:         
            validset = torchvision.datasets.CIFAR100(
                root=root, train=True, download=True, 
                transform=transform_train)
            valid_sampler = SubsetRandomSampler(valid_idx)    
        else:
            validset = torchvision.datasets.CIFAR100(
                root=root, train=True, download=True, 
                transform=transform_test)
            valid_sampler = SubsetSequantialSampler(valid_idx)                
        
        valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=valid_batch, sampler=valid_sampler,
            num_workers=workers, pin_memory=True)
        loaders["valid"] = valid_loader
    
    # training set
    if train_batch > 0:
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train)
        train_sampler = SubsetRandomSampler(train_idx)    
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch, sampler=train_sampler,
            num_workers=workers, pin_memory=True)
        loaders["train"] = train_loader

    # test set 
    if test_batch > 0:
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch, shuffle=False,
            num_workers=workers, pin_memory=True)
        loaders["test"] = test_loader

    return loaders

    