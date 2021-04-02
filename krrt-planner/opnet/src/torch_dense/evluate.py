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

import transform
import data_util
import scene_dataloader
# import model_dense_simple
# import model_dense
import model_dense_nosdf
import loss as loss_util

# python2 evluate.py --gpu 0 --model_path /home/wlz/catkin_ws/src/krrt-planner/opnet/models/nosdf_v0.pth --input_data_path /media/wlz/SSD/h3_80_40 --test_file_list ../filelists/5cm_80_40_val.txt  --output output/new_simple_aspp --max_to_vis 10

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--target_data_path', default='', help='path to target data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output', help='folder to output predictions')
parser.add_argument('--batch_size', type=int, default=10, help='input batch size')

# model params
parser.add_argument('--flip', dest='flipped', action='store_false')
parser.add_argument('--num_hierarchy_levels', type=int, default=2, help='#hierarchy levels.')
parser.add_argument('--max_input_height', type=int, default=128, help='max height in voxels')
parser.add_argument('--truncation', type=float, default=2, help='truncation in voxels')
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

# create model
model = model_dense_nosdf.GenModel(args.encoder_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense).cuda()
if not args.cpu:
    model = model.cuda()
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)


def test(loss_weights, dataloader, output_vis, num_to_vis):
    model.eval()
    val_losses = [ [] for i in range(args.num_hierarchy_levels+2) ]
    val_bceloss = []
    val_sdfloss = []
    val_precision = []
    val_recall = []
    val_f1score = []
    val_ious = []
    val_time = []
    num_vis = 0
    num_batches = len(dataloader)
    print("num_batches: ",num_batches)

    sigmoid_func = torch.nn.Sigmoid()
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            # print("sample = ",sample)
            sdfs = sample['sdf']
            inputs = sample['input']
            known = sample['known']
            hierarchy = sample['hierarchy']
            # print("check  hie=",hierarchy)
            for h in range(len(hierarchy)):
                hierarchy[h] = hierarchy[h].cuda()
                hierarchy[h].unsqueeze_(1)
            # if args.use_loss_masking:
            if True:
                known = known.cuda()
            inputs = inputs.cuda()
            sdfs = sdfs.cuda()
            target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs, hierarchy, 
                args.num_hierarchy_levels, args.truncation, True, known, flipped=args.flipped)
            # target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs, hierarchy, 
            #     1, args.truncation, True, known, flipped=args.flipped)

            start_time = time.time()
            output_sdf, output_occs = model(inputs, loss_weights)       
            known_mask = inputs > -1

            val_time.append(time.time() - start_time) 
            # loss, losses = loss_util.compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs[0], args.use_loss_masking, known)
            bce_loss, sdf_loss, losses, last_loss = loss_util.compute_loss_nosdf(output_sdf, output_occs, target_for_sdf, target_for_occs, 
                target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs,
                args.use_loss_masking, known, flipped=args.flipped)
            loss = bce_loss * 2 # + sdf_loss

            val_bceloss.append(last_loss)
            # val_sdfloss.append(sdf_loss)
            val_losses[0].append(loss)

            # output_sdf[known_mask] = inputs[known_mask]

            if output_sdf is not None:
                pred_occ = output_occs[-1][:,0].squeeze()
                pred_occ = sigmoid_func(pred_occ) > 0.0
                # pred_occ = output_sdf[:,1].squeeze()
                # pred_occ = (pred_occ).abs() < 2
                target_occ = target_for_occs[-1].squeeze()
                input_occ = inputs.squeeze()
                precision, recall, f1score = loss_util.compute_dense_occ_accuracy(input_occ, target_occ, pred_occ, truncation=args.truncation)
            else:
                precision, recall, f1score = 0,0,0

            val_precision.append(precision)
            val_recall.append(recall)
            val_f1score.append(f1score)
            val_losses[args.num_hierarchy_levels+1].append(losses[-1])

            if True:
                data_util.save_dense_predictions(args.output, sample['name'],
                     input_occ.cpu().numpy(), target_occ.cpu().numpy(), pred_occ.cpu().numpy(), args.truncation, flipped=args.flipped)
                print("saved: ", sample['name'])

    print("epoch_loss/total: ", np.mean(val_losses[0]))
    print("epoch_loss/bce: ", np.mean(val_bceloss))
    print("epoch_loss/precision: ", np.mean(val_precision))
    print("epoch_loss/recall: ", np.mean(val_recall))
    print("epoch_loss/f1: ", np.mean(val_f1score))
    print("average_time: ", np.mean(val_time))

    return


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

    test_dataset = scene_dataloader.DenseSceneDataset(test_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0, 0, flipped=args.flipped, 
                    # trans=transform.MyTransforms([transform.AddPepperNoise(0.90), 
                    #     transform.AddRandomFlip()], 
                    #     random_lift=transform.RandomliftFloor(), max_lift=2)
                    )
    # test_dataset = scene_dataloader.DenseSceneDataset(test_files, args.input_dim, args.truncation, 1, args.max_input_height, 0, args.target_data_path, test = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=scene_dataloader.collate_dense)

    # if os.path.exists(args.output):
    #     raw_input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # start testing
    print('starting testing...')
    loss_weights = torch.ones(args.num_hierarchy_levels+1).float()
    test(loss_weights, test_dataloader, args.output, args.max_to_vis)


if __name__ == '__main__':
    main()


