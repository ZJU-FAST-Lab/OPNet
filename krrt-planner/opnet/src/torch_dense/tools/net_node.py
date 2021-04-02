#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
# from voxblox_msgs.msg import CenteredPointcloud
from ocp_msgs.srv import PredictPCL, PredictPCLResponse
import argparse
from IPython import embed

import argparse
import os, sys
import torch
import numpy as np
import time

import data_util
import model_dense
import model_dense_simple
from test_model_nosurf import GenModel


# python2 net_node.py --model_path /home/wlz/vox_ws/src/opnet/models/h2-flip-14.pth --input_topic /voxblox_node/ctsdf_pointcloud --flipped --num_hierarchy_levels 2 --dimx 64 --dimz 48
# python2 net_node.py --model_path /home/wlz/catkin_ws/src/opnet/models/simple_aspp_punish.pth  --occ_thresord 1.0

# # params
# parser = argparse.ArgumentParser()
# # data paths
# parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
# parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

# # parser.add_argument('--model_path', required=True, help='path to model to test')
# # model params
# parser.add_argument('--num_hierarchy_levels', type=int, default=2, help='#hierarchy levels.')
# parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
# parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel_size in meters')
# parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
# parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
# parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')

# # test params
# # parser.add_argument('--max_to_vis', type=int, default=10, help='max num to vis')
# # parser.add_argument('--flipped', dest='flipped', action='store_true')
# parser.add_argument('--cpu', dest='cpu', action='store_true')
# parser.add_argument('--dimx', type=int, default=80, help='boundingbox dim x/y')
# parser.add_argument('--dimz', type=int, default=64, help='boundingbox dim z')
# parser.add_argument('--occ_thresord', type=float, default=1.0, help='thredsord value of occ and freespace')

# # ros params
# parser.add_argument('--service_name', type=str, default="/occ_map/pred", help='ros service name')


# parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
# args = parser.parse_args()
# assert( not (args.no_pass_feats or args.no_pass_occ) )
# assert( args.num_hierarchy_levels > 1 )
# args.input_nf = 1
# # print(args)

# # specify gpu
# os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

# # create model
# model = model_dense.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, True, True, True, True).cuda()
# # model = torch.jit.load('h2_nf_0917.zip')


# checkpoint = torch.load(args.model_path)
# model.load_state_dict(checkpoint['state_dict'])
# print('loaded model:', args.model_path)

# if not args.cpu:
#     model = model.cuda()
# model.eval()

# WEIGHTS = torch.ones(args.num_hierarchy_levels+1).float()

class ocp_net(object):

    def __init__(self):
        rospy.init_node('op_net', anonymous=True)
        self.init_param()
        self.weights = torch.ones(self.num_hierarchy_levels+1).float()
        
        # self.model = model_dense_simple.GenModel(8, 1, 16, 16, self.num_hierarchy_levels, True, True, True, True)
        self.model = GenModel(8, 1, 16, 16, 2, True, True, True, True)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('loaded model:', self.model_path)
        self.model.eval()
        if not self.use_cpu:
            self.model = self.model.cuda()

        self.server = rospy.Service(self.service_name, PredictPCL, self.handle_pred_pcl)
        self.sigmoid_func = torch.nn.Sigmoid()
        rospy.loginfo("net node ready!")
        rospy.spin()


    def init_param(self):
        self.use_cpu = rospy.get_param("/network/no_gpu", default=False)
        # self.model_path = rospy.get_param("/network/model_path", default="/home/wlz/catkin_ws/src/opnet/models/simple_aspp_punish.pth")
        self.model_path = rospy.get_param("/network/model_path", default='../../models/no_surf.pth')

        self.num_hierarchy_levels = rospy.get_param("/network/num_hierarchy_levels", default=2)
        self.occ_thresord = rospy.get_param("/network/occ_thresord", default=0.4)
        
        self.voxel_size = rospy.get_param("/network/voxel_size", default=0.05)
        
        self.service_name = rospy.get_param("/network/service_name", default="/occ_map/pred")

    def handle_pred_pcl(self, req):
        print("inference")
        if rospy.is_shutdown():
            exit()

        time1 = time.time()
        dimx = req.dim_x
        dimy = req.dim_y
        dimz = req.dim_z
        input = req.input
        input = np.array(req.input).astype(np.float32)
        input = torch.from_numpy(input).cuda().reshape([dimx, dimy, dimz]) #.unsqueeze(0)

        original_input = input
        # for grids whose tsdf value > -2, we consider it as already known
        known_mask = (original_input >= 0)

        original_occ = (original_input > 0)
        original_free = (original_input == 0)
        original_unkown = original_input < 0

        time2 = time.time()
        # print("prepocess time: %fs"%(time2 - time1))

        with torch.no_grad():
            input = input.unsqueeze(0)
            # output_sdf, _ = self.model(input, self.weights)
            output = self.model(input)

            # output_sdf = self.sigmoid_func(output_sdf[:,0])
            # sdf_to_occ = output_sdf > self.occ_thresord # < n (n>0, meaner when n get smaller)
            time3 = time.time()
            # print("model time: %fs"%(time3 - time2))

            # sdf_to_occ = output_sdf.abs() < self.occ_thresord # < n (n>0, meaner when n get smaller)
 
            # fix conflicts with input in known grids
            # sdf_to_occ.squeeze_()
            # print("output: ", output_sdf.shape)
            # sdf_to_occ[known_mask] = original_occ[known_mask]
            # sdf_to_occ = sdf_to_occ.float().cpu()
            output = output.cpu().numpy().reshape([-1]).tolist()
            # print("post process time: %fs"%(time.time() - time3))

        return PredictPCLResponse(output)
 

if __name__ == '__main__':
    ocp_net()