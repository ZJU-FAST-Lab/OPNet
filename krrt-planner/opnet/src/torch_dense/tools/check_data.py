from __future__ import division
from __future__ import print_function

import argparse
import os, sys, time
import shutil
import random
import torch
import numpy as np
import gc

import data_util
import scene_dataloader
# import model
import loss as loss_util


# python2 check_data.py --gpu 0  --data_path /media/wlz/4DB5D005268BA4D8/h3_80_size10_106  --train_file_list /home/wlz/opnet/data/h3_80_size10_106_train.txt  --save logs/test
# python2 train.py --gpu 0 --data_path ../data/611  --train_file_list ../filelists/611.txt --val_file_list ../filelists/test_val.txt --save logs/mp --num_hierarchy_levels 2

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data_path', required=True, help='path to data')
parser.add_argument('--train_file_list', required=True, help='path to file list of train data')
# parser.add_argument('--val_file_list', default='', help='path to file list of val data')
parser.add_argument('--save', default='./logs/test', help='folder to output model checkpoints')
# model params
# parser.add_argument('--retrain', type=str, default='', help='model to load from')
parser.add_argument('--input_dim', type=int, default=0, help='voxel dim.')

parser.add_argument('--num_hierarchy_levels', type=int, default=2, help='#hierarchy levels (must be > 1).')
# parser.add_argument('--num_iters_per_level', type=int, default=2000, help='#iters before fading in training for next level.')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

parser.add_argument('--vis_dfs', type=int, default=0, help='use df (iso 1) to visualize')
args = parser.parse_args()
# assert( not (args.no_pass_feats and args.no_pass_occ) )
# assert( args.weight_missing_geo >= 1)
parser.set_defaults(num_hierarchy_levels=4)

UP_AXIS = 0
print(args)
FLIP = False

train_files, val_files = data_util.get_train_files(args.data_path, args.train_file_list, None)

print('#train files = ', len(train_files))
train_dataset = scene_dataloader.DenseSceneDataset(train_files, [1,1,1], args.truncation, 2, 0, 0, flipped=FLIP)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=scene_dataloader.collate_dense, drop_last=True) # collate_fn=scene_dataloader.collate

_SPLITTER = ','

def check(dataloader, output_save):

    num_batches = len(dataloader)
    print("num_batches: ",num_batches)
    for t, sample in enumerate(dataloader):
        print("START")
        sdfs = sample['sdf']
        if sdfs.shape[0] < args.batch_size:
            print("empty sdf: ", sample['name'])
            continue  # maintain same batch size for training
        inputs = sample['input']
        known = sample['known']
        hierarchy = sample['hierarchy']

        target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs, hierarchy, args.num_hierarchy_levels, 
            args.truncation, True, known, flipped=FLIP)
        # print("target_for_occs: ", target_for_occs[-1].shape, (target_for_occs[-1]>0).sum(), (target_for_occs[-1]==0).sum())
        dims = sample['orig_dims'][0]
        inputs = inputs.cpu().numpy()

        # print(inputs[0].shape, inputs[0][0], target_for_sdf.shape)
        data_util.save_input_target_pred(args.save, sample['name'], inputs, 
            target_for_sdf, None, truncation=2)
            
        print("END")
    return

def main():
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    else:
        raw_input('warning: save dir %s exists, press key to delete and continue' % args.save)

    # data_util.dump_args_txt(args, os.path.join(args.save, 'args.txt'))
    # log_file = open(os.path.join(args.save, 'log.csv'), 'w')
    # headers = ['epoch','iter','train_loss(total)']
    # for h in range(args.num_hierarchy_levels):
    #     headers.append('train_loss(' + str(h) + ')')
    # headers.extend(['train_loss(sdf)', 'train_l1-pred', 'train_l1-tgt'])
    # for h in range(args.num_hierarchy_levels):
    #     headers.append('train_iou(' + str(h) + ')')
    # headers.extend(['time'])
    # log_file.write(_SPLITTER.join(headers) + '\n')
    # log_file.flush()

    # has_val = len(val_files) > 0
    # log_file_val = None
    # if has_val:
    #     headers = headers[:-1]
    #     headers.append('val_loss(total)')
    #     for h in range(args.num_hierarchy_levels):
    #         headers.append('val_loss(' + str(h) + ')')
    #     headers.extend(['val_loss(sdf)', 'val_l1-pred', 'val_l1-tgt'])
    #     for h in range(args.num_hierarchy_levels):
    #         headers.append('val_iou(' + str(h) + ')')
    #     headers.extend(['time'])
    #     log_file_val = open(os.path.join(args.save, 'log_val.csv'), 'w')
    #     log_file_val.write(_SPLITTER.join(headers) + '\n')
    #     log_file_val.flush()
    # start training
    # print('starting training...')
    # iter = args.start_epoch * (len(train_dataset) // args.batch_size)
    # for epoch in range(args.start_epoch, args.max_epoch):
        # start = time.time()


    check(train_dataloader, output_save=True)

if __name__ == '__main__':
    main()


