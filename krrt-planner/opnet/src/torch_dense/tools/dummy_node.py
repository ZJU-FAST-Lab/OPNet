#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
# from voxblox_msgs.msg import CenteredPointcloud
from occ_grid.srv import PredictPCL, PredictPCLResponse
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


class ocp_net(object):

    def __init__(self):
        rospy.init_node('opnet', anonymous=True)
        self.init_param()
        self.weights = torch.ones(self.num_hierarchy_levels+1).float()
        
        self.server = rospy.Service(self.service_name, PredictPCL, self.handle_pred_pcl)
        self.sigmoid_func = torch.nn.Sigmoid()
        rospy.loginfo("net node ready!")
        rospy.spin()


    def init_param(self):
        self.use_cpu = rospy.get_param("/network/no_gpu", default=False)
        # self.model_path = rospy.get_param("/network/model_path", default="/home/wlz/catkin_ws/src/opnet/models/simple_aspp_punish.pth")
        self.model_path = rospy.get_param("/network/model_path", default="/home/wlz/catkin_ws/src/opnet/models/simple_aspp.pth")

        self.num_hierarchy_levels = rospy.get_param("/network/num_hierarchy_levels", default=2)
        self.occ_thresord = rospy.get_param("/network/occ_thresord", default=0.5)
        
        self.voxel_size = rospy.get_param("/network/voxel_size", default=0.05)
        
        self.service_name = rospy.get_param("/network/service_name", default="/occ_map/pred")

    def handle_pred_pcl(self, req):
        time1 = time.time()
        dimx = req.dim_x
        dimy = req.dim_y
        dimz = req.dim_z
        input = req.input
        input = np.array(req.input).astype(np.float32)
        input = torch.from_numpy(input).cuda().reshape([dimx, dimy, dimz]) #.unsqueeze(0)

        original_input = input
        # print("prepocess time: %fs"%(time2 - time1))

        rospy.sleep(0.05)
        # only unknown
        sdf_to_occ = input < 0 
        sdf_to_occ = sdf_to_occ.float().reshape([-1]).cpu()
        sdf_to_occ = sdf_to_occ.numpy().tolist()
        # print("post process time: %fs"%(time.time() - time3))

        return PredictPCLResponse(sdf_to_occ)
 

if __name__ == '__main__':
    ocp_net()
