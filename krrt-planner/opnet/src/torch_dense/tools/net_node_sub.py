#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from voxblox_msgs.msg import CenteredPointcloud
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
import model_dense
import model_dense_simple

# python2 net_node.py --model_path /home/wlz/vox_ws/src/opnet/models/h2-flip-14.pth --input_topic /voxblox_node/ctsdf_pointcloud --flipped --num_hierarchy_levels 2 --dimx 64 --dimz 48
# python2 net_node_sub.py --model_path /home/wlz/vox_ws/src/opnet/models/h2_occ_sdf_simple.pth --input_topic /voxblox_node/ctsdf_pointcloud  --num_hierarchy_levels 2 --dimx 80 --dimz 48

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output', help='folder to output predictions')
# model params
parser.add_argument('--num_hierarchy_levels', type=int, default=3, help='#hierarchy levels.')
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
parser.add_argument('--occ_thresord', type=float, default=2.0, help='thredsord value of occ and freespace')

# ros params
parser.add_argument('--input_topic', type=str, default="/net_input", help='ros tsdf pointcloud input topic')
parser.add_argument('--output_topic', type=str, default="/net_output", help='ros tsdf pointcloud output topic')

parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
args = parser.parse_args()
assert( not (args.no_pass_feats or args.no_pass_occ) )
assert( args.num_hierarchy_levels > 1 )
args.input_nf = 1
# print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

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

# create model
model = model_dense_simple.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, True, True, True, True).cuda()
# model = torch.jit.load('h2_nf_0917.zip')


checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)

if not args.cpu:
    model = model.cuda()
model.eval()

# WEIGHTS = torch.from_numpy(np.ones(args.num_hierarchy_levels+1, dtype=np.float32))
WEIGHTS = torch.ones(args.num_hierarchy_levels+1).float()
occ_pub = rospy.Publisher(args.output_topic, PointCloud2, queue_size=3)
added_occ_pub = rospy.Publisher(args.output_topic + "_added", PointCloud2, queue_size=3)

# input_dim = np.array([args.dimz, args.dimx, args.dimx])
hierarchy_factor = pow(2, args.num_hierarchy_levels-1)
# model.update_sizes(input_dim, input_dim // hierarchy_factor)



def tsdf_callback(data):
    # gen = point_cloud2.read_points(data, skip_nans=True)
    # print(type(gen))
    # exit()
    time1 = time.time()
    gen = point_cloud2.read_points(data.points, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    center = data.center

    points = np.array(list(gen)).reshape([-1, 4])
    print(points.shape, points[:, -1].max())

    offset_x = center[0] - args.voxel_size * args.dimx / 2
    offset_y = center[1] - args.voxel_size * args.dimx / 2
    offset_z = - 2.7 # points[:, 2].min() -0.1

    points[:, 0] -= offset_x
    points[:, 1] -= offset_y
    points[:, 2] -= offset_z
    # print("minz: ", offset_z)

    points = points / args.voxel_size
    locs = np.floor(points[:, :3]).astype(np.int32)
    feats = points[:, 3]

    # print("feats = ",feats)
    mask = (locs[:, 0] < args.dimx)  & (locs[:, 1] < args.dimx) & (locs[:, 2] < args.dimz) & (locs[:, 0] >= 0) & (locs[:, 1] >= 0) & (locs[:, 2] >= 0)
    locs = locs[mask]
    feats = feats[mask]
    input = data_util.sparse_to_dense_np(locs, feats, args.dimx, args.dimx, args.dimz, -float('inf'))
    input = torch.from_numpy(input) #.unsqueeze(0)
    # print("shape after s to d", input.shape)

    original_input = input
    print("original_input: ", original_input.shape)
    # for grids whose tsdf value > -2, we consider it as already known
    known_mask = (original_input > -2)

    original_occ = (original_input.abs() < args.occ_thresord)
    original_free = original_input > args.occ_thresord
    original_unkown = original_input < - 2
    # print("INPUT: %f  occ, %f  free, %f  unkwn"%(original_occ.float().sum()/SUM, original_free.float().sum()/SUM, original_unkown.float().sum()/SUM))

    if args.flipped:
        input[input.abs() > args.truncation] = 0
        input = torch.sign(input) * (args.truncation - input.abs())
    else:
        input[input > args.truncation] = args.truncation
        input[input < -args.truncation] = -args.truncation

    # print("input mean:",input.abs().mean(), input.mean(), "  max=",input.max())
    # print("known: %f %%"%(known_mask.float().sum() * 100 /(64*64*48)))


    time2 = time.time()
    print("prepocess time: %fs"%(time2 - time1))

    with torch.no_grad():
        # print("input_dim: ", np.array(locs.shape))
        input = input.unsqueeze(0)
        if not args.cpu:
            input = input.cuda()
        
        output_occs = None

        # try:
        output_sdf, output_occs = model(input, WEIGHTS)
        # print("out mean: ", output_sdf.abs().mean())

        # except:
        #     print("MODEL FAIL")
        #     # embed()
        # output_sdf = output_sdf/args.voxel_size

        # fix conflicts with input in known grids
        known_mask=known_mask.unsqueeze(0).unsqueeze(0)
        output_sdf[known_mask] = input[known_mask]

        if not args.flipped:
            sdf_to_occ = output_sdf.abs() < args.occ_thresord # < n (n>0, meaner when n get smaller)
        else:
            sdf_to_occ = output_sdf.abs() > args.occ_thresord  # > n (n<3, meaner when n get bigger)

        sdf_to_occ = sdf_to_occ.cpu()

        # wlz: original occ:input, sdf_to_occ:output, added_occ:added
        sdf_to_occ = sdf_to_occ.squeeze(0).squeeze(0)
        # sdf_to_occ = output_occs[-1][0].cpu().squeeze(0)
        # sdf_to_occ = torch.sigmoid(sdf_to_occ) > 0.5
        added_occ = sdf_to_occ.float() > original_occ.float()

        time3 = time.time()
        print("model time: %fs"%(time3 - time2))

        if sdf_to_occ is not None:
            # # output_occs = output_occs.cpu().squeeze()
            # original_occ = original_occ.squeeze()
            # occ_coords = POINTER[original_occ].reshape([-1, 3])
            # num_points = occ_coords.shape[0]
            # added_color = torch.ones(num_points).unsqueeze(1).cuda() * (-0.2)
            # occ_coords = torch.cat((occ_coords, added_color), 1)
            # print("original occ size: ",occ_coords.shape)

            added_occ = added_occ.squeeze()
            occ_coords_1 = POINTER[added_occ].reshape([-1, 3])
            num_points = occ_coords_1.shape[0]
            added_color_1 = torch.ones(num_points).unsqueeze(1).cuda() * 10
            occ_coords_1 = torch.cat((occ_coords_1, added_color_1), 1)
            # print("added size: ",occ_coords_1.shape)    

            # occ_coords = torch.cat((occ_coords, occ_coords_1), 0) 
            occ_coords = occ_coords_1

            if occ_coords.shape[0] > 0:
                occ_coords *= args.voxel_size
                occ_coords[:, 0] += offset_x
                occ_coords[:, 1] += offset_y
                occ_coords[:, 2] += offset_z
                occ_coords = occ_coords.cpu().numpy()

                msg = data.points
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
                # print(occ_coords.shape, len(msg.data), msg.row_step)
                # print("postpocess time: %fs"%(time.time() - time3))
                publish_pcl(msg)
            
            #publish added occ
            # original_unkown = original_input <= -2
            original_unkown = original_unkown.squeeze()
            occ_coords = POINTER[original_occ].reshape([-1, 3])
            # print("known occ size: ",occ_coords.shape)

            if occ_coords.shape[0] > 0:
                occ_coords *= args.voxel_size
                occ_coords[:, 0] += offset_x
                occ_coords[:, 1] += offset_y
                occ_coords[:, 2] += offset_z
                occ_coords = occ_coords.cpu().numpy()

                msg = data.points
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
    return


def publish_pcl(msg):
    occ_pub.publish(msg)
    return

def publish_added_pcl(msg):
    added_occ_pub.publish(msg)
    return    


if __name__ == '__main__':
    use_cuda = True
    rospy.init_node('opnet_net', anonymous=True)
    rospy.Subscriber(args.input_topic, CenteredPointcloud, tsdf_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()