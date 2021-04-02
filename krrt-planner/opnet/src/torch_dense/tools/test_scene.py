from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import random
import torch
import numpy as np
import gc
import time
from IPython import embed

import data_util
import scene_dataloader
import model_dense
import loss as loss_util

# python2 test_scene.py --gpu 0 --model_path ./logs/h3_64/model-iter14000-epoch35.pth --input_data_path ../data/data616  --test_file_list ../filelists/data616_train.txt --output output/h2_64f --max_to_vis 10 --num_hierarchy_levels 3
# python2 test_scene.py --gpu 0 --model_path ./logs/h2_64/model-epoch-5.pth --input_data_path /media/wlz/Elements/datagen/GenerateScans/output/h3_64  --test_file_list ../filelists/h3_64_train.txt --output output/mp/h3_64_train --max_to_vis 5

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--target_data_path', default='', help='path to target data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output', help='folder to output predictions')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

# model params
parser.add_argument('--flip', dest='flipped', action='store_true')
parser.add_argument('--num_hierarchy_levels', type=int, default=2, help='#hierarchy levels.')
parser.add_argument('--max_input_height', type=int, default=128, help='max height in voxels')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=128, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
# test params
parser.add_argument('--max_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--cpu', dest='cpu', action='store_true')


parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.num_hierarchy_levels > 1 )
args.input_nf = 1
# print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
UP_AXIS = 0 # z is 0th 


# create model
model = model_dense.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense).cuda()
if not args.cpu:
    model = model.cuda()
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)


def test(loss_weights, dataloader, output_vis, num_to_vis):
    model.eval()

    num_vis = 0
    num_batches = len(dataloader)
    print("num_batches: ",num_batches)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            inputs = sample['input']
            print("input_dim: ", np.array(sample['sdf'].shape))
            input_dim = np.array(sample['sdf'].shape[2:])

            print(input_dim)
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (num_vis, num_to_vis, sample['name'], input_dim[0], input_dim[1], input_dim[2]))
            sys.stdout.flush()

            # hierarchy_factor = pow(2, args.num_hierarchy_levels-1)
            # model.update_sizes(input_dim, input_dim // hierarchy_factor)
            
            sdfs = sample['sdf']
            if sdfs.shape[0] < args.batch_size:
                continue  # maintain same batch size for training
            inputs = sample['input']
            known = sample['known']
            hierarchy = sample['hierarchy']
            for h in range(len(hierarchy)):
                hierarchy[h] = hierarchy[h].cuda()
                hierarchy[h].unsqueeze_(1)
            if args.use_loss_masking:
                known = known.cuda()
            inputs = inputs.cuda()
            sdfs = sdfs.cuda()
            target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs, hierarchy, 
                args.num_hierarchy_levels, args.truncation, args.use_loss_masking, known, flipped=args.flipped)

            start_time = time.time()
            try:
                output_sdf, output_occs = model(inputs, loss_weights)
            except:
                print('exception at %s' % sample['name'])
                gc.collect()
                continue

            end_time = time.time()
            print('TIME: %.4fs'%(end_time - start_time))

            # remove padding
            # dims = sample['orig_dims'][0]
            # mask = (output_sdf[0][:,0] < dims[0]) & (output_sdf[0][:,1] < dims[1]) & (output_sdf[0][:,2] < dims[2])
            # output_sdf[0] = output_sdf[0][mask]
            # output_sdf[1] = output_sdf[1][mask]
            # mask = (inputs[0][:,0] < dims[0]) & (inputs[0][:,1] < dims[1]) & (inputs[0][:,2] < dims[2])
            # inputs[0] = inputs[0][mask]
            # inputs[1] = inputs[1][mask]
            # # vis_pred_sdf = [None]
            # # if len(output_sdf[0]) > 0:
            # #     vis_pred_sdf[0] = [output_sdf[0].cpu().numpy(), output_sdf[1].squeeze().cpu().numpy()]
            # inputs = [inputs[0].cpu().numpy(), inputs[1].cpu().numpy()]
            
            # vis occ & sdf
            pred_occs = [None] * args.num_hierarchy_levels
            for h in range(args.num_hierarchy_levels):
                factor = 2**(args.num_hierarchy_levels-h-1)
                pred_occs[h] = [None] * args.batch_size
                if len(output_occs[h][0]) == 0:
                    continue
                # filter: occ > 0 
                for b in range(args.batch_size):
                    batchmask = output_occs[h][0][:,-1] == b
                    locs = output_occs[h][0][batchmask][:,:-1]
                    vals = output_occs[h][1][batchmask]
                    occ_mask = torch.nn.Sigmoid()(vals[:,0].detach()) > 0.5
                    if args.flipped:
                        pred_occs[h][b] = [locs[occ_mask.view(-1)].cpu().numpy(), vals[occ_mask.view(-1)].cpu().numpy()]
                    else:
                        pred_occs[h][b] = locs[occ_mask.view(-1)].cpu().numpy()

            vis_pred_sdf = [None] * args.batch_size
            if len(output_sdf[0]) > 0:
                for b in range(args.batch_size):
                    mask = output_sdf[0][:,-1] == b
                    if len(mask) > 0:
                        vis_pred_sdf[b] = [output_sdf[0][mask].cpu().numpy(), output_sdf[1][mask].squeeze().cpu().numpy()]

            data_util.save_predictions(output_vis, sample['name'], inputs, target_for_sdf.cpu().numpy(), None, vis_pred_sdf, pred_occs, 
                sample['world2grid'], args.truncation, flipped=args.flipped)
            num_vis += 1
            if num_vis >= num_to_vis:
                break
    sys.stdout.write('\n')


def main():
    # data files
    test_files, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '')
    if len(test_files) > args.max_to_vis:
        test_files = test_files[:args.max_to_vis]
    else:
        args.max_to_vis = len(test_files)
    random.seed(42)
    random.shuffle(test_files)
    print('#test files = ', len(test_files))
    # test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, args.num_hierarchy_levels, args.max_input_height, 0, args.target_data_path)
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, 1, args.max_input_height, 0, args.target_data_path, test = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=scene_dataloader.collate)

    if os.path.exists(args.output):
        raw_input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # start testing
    print('starting testing...')
    loss_weights = torch.from_numpy(np.ones(args.num_hierarchy_levels+1, dtype=np.float32))
    test(loss_weights, test_dataloader, args.output, args.max_to_vis)


if __name__ == '__main__':
    main()


