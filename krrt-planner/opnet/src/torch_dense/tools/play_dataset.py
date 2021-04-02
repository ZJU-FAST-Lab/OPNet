#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
# from voxblox_msgs.msg import Densesdf
import argparse
from IPython import embed

import argparse
import os, sys
import random
import torch
import numpy as np
import gc
import time

import data_util
import scene_dataloader
import model_dense

# python2 net_node.py --model_path /home/wlz/vox_ws/src/opnet/models/flip_epoch16_iter34000.pth --input_topic /voxblox_node/tsdf_pointcloud --flipped --dimx 80 --dimz 64
# python2 net_node.py --model_path /home/wlz/vox_ws/src/opnet/models/model-iter28000-epoch13.pth --input_topic /voxblox_node/tsdf_pointcloud  --dimx 80 --dimz 64

# python2 net_node.py --model_path /home/pn/wlz_ws/forward_opnet/models/noflip_h3_epoch18.pth --input_topic /voxblox_node/tsdf_pointcloud --num_hierarchy_levels 3 --dimx 200 --dimz 96 --voxel_size 0.02


# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')

# model params
parser.add_argument('--num_hierarchy_levels', type=int, default=2, help='#hierarchy levels.')
parser.add_argument('--max_input_height', type=int, default=128, help='max height in voxels')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel_size in meters')
parser.add_argument('--input_dim', type=int, default=128, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
# test params
# parser.add_argument('--max_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--flipped', dest='flipped', action='store_true')
parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.add_argument('--dimx', type=int, default=80, help='boundingbox dim x/y')
parser.add_argument('--dimz', type=int, default=64, help='boundingbox dim z')
# ros params
parser.add_argument('--input_topic', type=str, default="/net_input", help='ros tsdf pointcloud input topic')
parser.add_argument('--output_topic', type=str, default="/net_output", help='ros tsdf pointcloud output topic')

parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.num_hierarchy_levels > 1 )
args.input_nf = 1
# print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
UP_AXIS = 0 # z is 0th 
POINTER = torch.zeros((args.dimx, args.dimx, args.dimz, 3)).float()
for x in range(POINTER.shape[0]):
    POINTER[x, :, :, 0] = x
for y in range(POINTER.shape[1]):
    POINTER[:, y, :, 1] = y
for z in range(POINTER.shape[2]):
    POINTER[:, :, z, 2] = z

SUM = args.dimx * args.dimx * args.dimz

if not args.cpu:
    POINTER = POINTER.cuda()

test_files, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '')
if len(test_files) > args.max_to_vis:
    test_files = test_files[:args.max_to_vis]
else:
    args.max_to_vis = len(test_files)

print('#test files = ', len(test_files))
test_dataset = scene_dataloader.DenseSceneDataset(test_files, args.input_dim, args.truncation, 1, args.max_input_height, 0, args.target_data_path, test = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, flipped=False, collate_fn=scene_dataloader.collate_dense)


# create model
input_dim = np.array([args.dimz, args.dimx, args.dimx])
model = model_dense.GenModel(args.encoder_dim, input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense).cuda()
if not args.cpu:
    model = model.cuda()
print('loaded model:', args.model_path)
model.eval()

# WEIGHTS = torch.from_numpy(np.ones(args.num_hierarchy_levels+1, dtype=np.float32))
WEIGHTS = torch.from_numpy(np.ones(args.num_hierarchy_levels+1, dtype=np.float32))

occ_pub = rospy.Publisher(args.output_topic, PointCloud2, queue_size=3)
added_occ_pub = rospy.Publisher(args.output_topic + "_added", PointCloud2, queue_size=3)


hierarchy_factor = pow(2, args.num_hierarchy_levels-1)
# model.update_sizes(input_dim, input_dim // hierarchy_factor)


def play_dataset(dataloader):
    for t, data in enumerate(dataloader):
        time1 = time.time()
        trunc = args.truncation * args.voxel_size

        offset_x = 0
        offset_y = 0
        offset_z = 0

        inputs = sample['input']
        if not args.cpu:
            inputs = inputs.cuda()
        # print("shape after s to d", input.shape)
        input = inputs.squeeze()
        original_input = input
        # for grids whose tsdf value > 2, we consider it as already known
        known_mask = (original_input > -2)

        original_occ = (original_input.abs() < 2)
        original_free = original_input > 2
        original_unkown = original_input < - 2
        # print("INPUT: %f  occ, %f  free, %f  unkwn"%(original_occ.float().sum()/SUM, original_free.float().sum()/SUM, original_unkown.float().sum()/SUM))

        if args.flipped:
            input[input.abs() > args.truncation] = 0
            # print("1", input.shape)
            input = torch.sign(input) * (args.truncation - input.abs())
            # print("2", input.shape)
        else:
            input[input > args.truncation] = args.truncation
            input[input < -args.truncation] = -args.truncation

        print("mean=",input.mean(),"   max=",input.max())


        time2 = time.time()
        print("prepocess time: %fs"%(time2 - time1))

        with torch.no_grad():
            # print("input_dim: ", np.array(locs.shape))
            input = input.unsqueeze(0)
            if not args.cpu:
                input = input.cuda()
            
            output_occs = None
            print("3", input.shape)
            print("weight.shape = ",WEIGHTS.shape)
            # try:
            output_sdf, output_occs = model(input, WEIGHTS)
            # except:
            #     print("MODEL FAIL")
            #     # embed()
            output_sdf=output_sdf/args.voxel_size
            #? wlz: wrong 
            if not args.flipped:
                sdf_to_occ = output_sdf.abs() < 1.0 # < n (n>0, meaner when n get smaller)
            else:
                sdf_to_occ = output_sdf.abs() > 0.5  # > n (n<3, meaner when n get bigger)

            # fix conflicts with input in known grids
            sdf_to_occ = sdf_to_occ.cpu()
            # known_mask=known_mask.unsqueeze(0)

            # wlz: original occ:input, sdf_to_occ:output, added_occ:added
            sdf_to_occ = sdf_to_occ.squeeze(0).squeeze(0)
            added_occ = sdf_to_occ.float() > original_occ.float()
            sdf_to_occ = original_occ
            # added_occ = original_unkown

            time3 = time.time()
            print("model time: %fs"%(time3 - time2))

            # out_locs = []
            # print(output_occs.shape)
            
            # wlz: try this(optimized)
            # sdf_to_occ = input == -3
            if sdf_to_occ is not None:
                # output_occs = output_occs.cpu().squeeze()
                sdf_to_occ = sdf_to_occ.squeeze()
                occ_coords = POINTER[sdf_to_occ].reshape([-1, 3])
                num_points = occ_coords.shape[0]
                added_color = torch.ones(num_points).unsqueeze(1).cuda()
                occ_coords = torch.cat((occ_coords, added_color), 1)
                print("occ_coords: ",occ_coords.shape)

                added_occ = added_occ.squeeze()
                occ_coords_1 = POINTER[added_occ].reshape([-1, 3])
                num_points = occ_coords_1.shape[0]
                added_color_1 = torch.ones(num_points).unsqueeze(1).cuda() * 5
                occ_coords_1 = torch.cat((occ_coords_1, added_color_1), 1)
                print("added_coords: ",occ_coords_1.shape)    

                occ_coords = torch.cat((occ_coords, occ_coords_1), 0)            

                if occ_coords.shape[0] > 0:
                    occ_coords *= args.voxel_size
                    occ_coords[:, 0] += offset_x
                    occ_coords[:, 1] += offset_y
                    occ_coords[:, 2] += offset_z
                    occ_coords = occ_coords.cpu().numpy()

                    msg = PointCloud2()
                    msg.height = 1
                    msg.width = occ_coords.shape[0]
                    msg.fields = [
                        PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('intensity', 12, PointField.FLOAT32, 1)]
                    msg.is_bigendian = False
                    msg.point_step = 16 #12
                    msg.row_step = 16 * occ_coords.shape[0]
                    msg.is_dense = int(np.isfinite(occ_coords).all())
                    msg.data = occ_coords.tostring() #np.asarray(occ_coords, np.float32).tostring()
                    # print(out_locs.shape, len(msg.data), msg.row_step)
                    # print("postpocess time: %fs"%(time.time() - time3))
                    publish_pcl(msg)
                
                #publish added occ
                added_occ = added_occ.squeeze()
                occ_coords = POINTER[added_occ].reshape([-1, 3])
                print("added_coords: ",occ_coords.shape)

                if occ_coords.shape[0] > 0:
                    occ_coords *= args.voxel_size
                    occ_coords[:, 0] += offset_x
                    occ_coords[:, 1] += offset_y
                    occ_coords[:, 2] += offset_z
                    occ_coords = occ_coords.cpu().numpy()

                    msg = data
                    msg.height = 1
                    msg.width = occ_coords.shape[0]
                    msg.fields = [
                        PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),]
                    msg.is_bigendian = False
                    msg.point_step = 12 #12
                    msg.row_step = 12 * occ_coords.shape[0]
                    msg.is_dense = int(np.isfinite(occ_coords).all())
                    msg.data = occ_coords.tostring() #np.asarray(occ_coords, np.float32).tostring()
                    publish_added_pcl(msg)            
        rospy.sleep(0.1)


def publish_pcl(msg):
    occ_pub.publish(msg)
    return

def publish_added_pcl(msg):
    added_occ_pub.publish(msg)
    return    


if __name__ == '__main__':
    use_cuda = True
    rospy.init_node('opnet', anonymous=True)
    rospy.Subscriber(args.input_topic, PointCloud2, tsdf_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
