import os, sys, struct
import numpy as np
import torch
import torch.utils.data
import math
import plyfile
import data_util

def collate(batch):
    names = [x['name'] for x in batch]
    # collect sparse inputs
    locs = batch[0]['input'][0]
    locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).long()], 1)
    feats = batch[0]['input'][1]
    known = None
    if batch[0]['known'] is not None:
        known = torch.stack([x['known'] for x in batch])
    colors = None
    hierarchy = None
    if batch[0]['hierarchy'] is not None:
        hierarchy = [None]*len(batch[0]['hierarchy'])
        for h in range(len(batch[0]['hierarchy'])):
            hierarchy[h] = torch.stack([x['hierarchy'][h] for x in batch])
    for b in range(1, len(batch)):
        cur_locs = batch[b]['input'][0]
        cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long()*b], 1)
        locs = torch.cat([locs, cur_locs])
        feats = torch.cat([feats, batch[b]['input'][1]])
    sdfs = torch.stack([x['sdf'] for x in batch])
    world2grids = torch.stack([x['world2grid'] for x in batch])
    orig_dims = torch.stack([x['orig_dims'] for x in batch])
    return {'name': names, 'input': [locs,feats], 'sdf': sdfs, 'world2grid': world2grids, 'known': known, 'hierarchy': hierarchy, 'orig_dims': orig_dims}

def collate_dense(batch):
    names = [x['name'] for x in batch]
    known = None
    if batch[0]['known'] is not None:
        known = torch.stack([x['known'] for x in batch])
        known.unsqueeze_(1)
    colors = None
    hierarchy = None
    if batch[0]['hierarchy'] is not None:
        hierarchy = [None]*len(batch[0]['hierarchy'])
        for h in range(len(batch[0]['hierarchy'])):
            hierarchy[h] = torch.stack([x['hierarchy'][h] for x in batch])

    input = torch.stack([x['input'] for x in batch])
    target = torch.stack([x['sdf'] for x in batch])
    world2grids = torch.stack([x['world2grid'] for x in batch])
    orig_dims = torch.stack([x['orig_dims'] for x in batch])

    return {'name': names, 'input': input, 'sdf': target, 'world2grid': world2grids, 'known': known, 'hierarchy': hierarchy, 'orig_dims': orig_dims}


class DenseSceneDataset(torch.utils.data.Dataset):

    def __init__(self, files, truncation, num_hierarchy_levels, target_path='',  trans=None):
        assert(num_hierarchy_levels <= 4) # havent' precomputed more than this
        self.is_chunks = target_path == '' # have target path -> full scene data
        if not target_path:
            self.files = [f for f in files if os.path.isfile(f)]
        else:
            self.files = [(f,os.path.join(target_path, os.path.basename(f))) for f in files if (os.path.isfile(f) and os.path.isfile(os.path.join(target_path, os.path.basename(f))))]
        self.truncation = truncation
        self.num_hierarchy_levels = num_hierarchy_levels
        self.UP_AXIS = 0
        self.flipped = True
        self.is_chunks = True

        self.transform = trans

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        name = None
        if self.is_chunks:
            name = os.path.splitext(os.path.basename(file))[0]
            
            inputs, targets, dims, world2grid, target_known, target_hierarchy = data_util.load_dense_train_file(file, self.num_hierarchy_levels)
        else:
            raise ValueError("dataloader")
        
        orig_dims = torch.LongTensor(targets.shape)

        # else:
        #     # need to change?
        #     if self.num_hierarchy_levels < 4:
        #         target_hierarchy = target_hierarchy[4-self.num_hierarchy_levels:]

        # obstacle & unknown space

        #transform
        if self.transform:
            # print("input: ", inputs.shape)
            # print("targets: ", targets.shape)
            inputs, targets, target_hierarchy = self.transform(inputs, targets, target_hierarchy)

        inputs[inputs > self.truncation] = self.truncation
        inputs[inputs < -self.truncation] = -self.truncation
        targets[targets > self.truncation] = self.truncation
        targets[targets < -self.truncation] = -self.truncation

        if self.flipped:
            inputs = data_util.tsdf_to_bool(inputs, trunc=self.truncation)

        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        
        targets = targets[np.newaxis,:]
        
        if target_hierarchy is not None:
            for h in range(len(target_hierarchy)):
                target_hierarchy[h] = torch.from_numpy(target_hierarchy[h]).float()
                target_hierarchy[h][target_hierarchy[h] > self.truncation] = self.truncation
                target_hierarchy[h][target_hierarchy[h] < -self.truncation] = -self.truncation

        world2grid = torch.from_numpy(world2grid)
        target_known = torch.from_numpy(target_known)

        sample = {'name': name, 'input': inputs, 'sdf': targets, 'world2grid': world2grid, 'known': target_known, 'hierarchy': target_hierarchy, 'orig_dims': orig_dims}
        return sample




