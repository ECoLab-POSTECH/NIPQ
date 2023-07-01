from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_loader(root, test_batch, train_batch, valid_size=0, cutout=16, 
    num_workers=4, download=True, random_seed=12345, shuffle=False):
    
    indices = list(range(50000))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
            
    train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
    
    if train_batch > 0:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4913999, 0.48215866, 0.44653133),
                (0.24703476, 0.24348757, 0.26159027)
            )
        ])
        
        if cutout > 0:
            transform_train.transforms.append(Cutout(cutout))

        train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True,
            download=download, transform=transform_train)
        train_sampler = SubsetRandomSampler(train_idx)      
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True)
           
    else:
        train_loader = None

    if valid_size > 0:
        assert(test_batch > 0, "validation set follows the batch size of test set, which is 0")
        
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4913999, 0.48215866, 0.44653133),
                (0.24703476, 0.24348757, 0.26159027)),
        ])    
        
        valid_dataset = datasets.CIFAR10(
            root=root, train=True,
            download=download, transform=transform_valid)        
        valid_sampler = SubsetRandomSampler(valid_idx)
         
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=test_batch, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=False, drop_last=False)
    else:
        valid_loader = None    
    
    if test_batch > 0:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4913999, 0.48215866, 0.44653133),
                (0.24703476, 0.24348757, 0.26159027)),
        ])

        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=test_batch, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        test_loader = None

    return test_loader, train_loader, valid_loader
