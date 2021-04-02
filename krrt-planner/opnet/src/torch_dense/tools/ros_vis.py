import os, sys, time
from copy import deepcopy
import random
import torch
import numpy as np

import data_util
import scene_dataloader
# import model_dense
# import model_dense_simple
import model_dense_nosurf
# import model_dense_sgnn
import loss as loss_util

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from IPython import embed

class ocp_net(object):
    
    def __init__(self):
        rospy.init_node('net_visualizer', anonymous=True)
        self.init_param()
        
        self.size_init = False
        # self.model = model_dense_simple.GenModel(8, 1, 16, 16, self.num_hierarchy_levels, True, True, True, True)
        self.model = model_dense_nosurf.GenModel(8, 1, 16, 16, self.num_hierarchy_levels, True, True, True, True).cuda()
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('loaded model:', self.model_path)
        self.model.eval()
        if not self.use_cpu:
            self.model = self.model.cuda()
        self.sigmoid_func = torch.nn.Sigmoid()
        rospy.loginfo("net node ready!")

        val_files, _ = data_util.get_train_files(self.data_path, self.file_list, '')
        print('#val files = ', len(val_files))
        if len(val_files) > 0:
            val_dataset = scene_dataloader.DenseSceneDataset(val_files, 2.0, 2)
            self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=scene_dataloader.collate_dense, drop_last=True) # collate_fn=scene_dataloader.collate

    def init_param(self):
        
        self.delay = int(10.0)
        self.rate = rospy.Rate(1)
        self.use_cpu = rospy.get_param("/network/no_gpu", default=False)
        self.model_path = rospy.get_param("/network/model_path", default="/home/wlz/catkin_ws/src/krrt-planner/opnet/models/no_surf.pth")
        # self.data_path = rospy.get_param("/network/data_path", default="/media/wlz/SSD/h3_80_40")
        self.data_path = rospy.get_param("/network/data_path", default="/media/wlz/Elements/datagen/GenerateScans/output/h3_80_40")

        self.file_list = rospy.get_param("/network/file_list", default="/home/wlz/catkin_ws/src/krrt-planner/opnet/src/filelists/name.txt")
        # self.file_list = rospy.get_param("/network/file_list", default="/home/wlz/catkin_ws/src/opnet/data/test.txt")

        self.num_hierarchy_levels = rospy.get_param("/network/num_hierarchy_levels", default=2)
        self.occ_thresord = rospy.get_param("/network/occ_thresord", default=1.0)
        
        self.voxel_size = rospy.get_param("/network/voxel_size", default=0.05)
        # topic that output pcl
        self.output_topic = rospy.get_param("/network/output_topic", default="/nework/output")
        self.output_pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=3)

        self.weights = torch.ones(self.num_hierarchy_levels+1).float()
        self.max_num = 1000

    def vis(self):
        count = 0
        with torch.no_grad():
            for t, sample in enumerate(self.val_dataloader):
                if rospy.is_shutdown():
                    exit()
                if count > self.max_num:
                    return
                else:
                    count += 1

                print("name: ", sample['name'])
                sdfs = sample['sdf']
                inputs = sample['input']
                known = sample['known']
                hierarchy = sample['hierarchy']
                for h in range(len(hierarchy)):
                    hierarchy[h] = hierarchy[h].cuda()
                    hierarchy[h].unsqueeze_(1)
                known = known.cuda()
                inputs = inputs.cuda()
                sdfs = sdfs.cuda()
                predz = int(np.floor(inputs.shape[-1] / 8) * 8)
                # embed()
                inputs = inputs[:,:,:,:predz]
                sdfs = sdfs[:,:,:,:,:predz]
                # embed()
                target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs, hierarchy, 
                    self.num_hierarchy_levels, 2.0, True, known, True)
                # embed()   

                output_sdf, output_occs = self.model(inputs, self.weights)     
                # bce_loss, sdf_loss, losses, last_loss = loss_util.compute_loss_dense(output_sdf, output_occs, target_for_sdf, target_for_occs, 
                #     target_for_hier, self.weights, 3.0, True, 3.0, inputs,
                #     True, known, flipped=True)
                # loss = bce_loss + sdf_loss

                # val_bceloss.append(last_loss)
                # val_sdfloss.append(sdf_loss)
                # val_losses[0].append(loss)

                # output_sdf[known_mask] = inputs[known_mask]

                if output_occs is not None:
                    start_time = time.time()
                    # pred_occ = output_sdf[:,0].squeeze()
                    pred_occ = output_occs[-1][:,0].squeeze()
                    pred_occ = self.sigmoid_func(pred_occ) > 0.45
                    # pred_occ = output_sdf[:,1].squeeze()
                    # pred_occ = (pred_occ).abs() < 2
                    target_occ = target_for_occs[-1].squeeze() > 0
                    input_occ = inputs.squeeze() > 0
                    # embed()
                    # precision, recall, f1score = loss_util.compute_dense_occ_accuracy(input_occ, target_occ, pred_occ, truncation=3)
                    
                    dimx = pred_occ.shape[-3]
                    dimy = pred_occ.shape[-2]
                    dimz = pred_occ.shape[-1]
                    # if not self.size_init:
                    self.init_size(dimx, dimy, dimz)

                    known_mask = (input_occ >= 0)
                    # fix conflicts with input in known grids
                    # pred_occ[known_mask] = input_occ[known_mask]

                    # num = (input_occ | pred_occ | target_occ).float().sum().cpu().numpy().tolist()
                    # print("total size: ", num)

                    # b - r - y
                    # vis pub
                    # embed()
                    input_occ_coords = self.pointers[input_occ]
                    input_colors = torch.FloatTensor([255,50,50]).repeat([input_occ_coords.shape[0], 1])
                    input_points = torch.cat([input_occ_coords, input_colors], dim=1)
                    input_points[:, :3] *= self.voxel_size
                    # print("input size: ", input_points.shape)


                    # red
                    pred_occ_coords = self.pointers[pred_occ & (~input_occ)]
                    pred_colors = torch.FloatTensor([50,50,255]).repeat([pred_occ_coords.shape[0], 1])
                    pred_points = torch.cat([pred_occ_coords, pred_colors], dim=1)
                    pred_points[:, :3] *= self.voxel_size
                    # print("pred size: ", pred_points.shape)

                    target_occ_coords = self.pointers[target_occ & (~input_occ)] # &(~pred_occ)
                    target_colors = torch.FloatTensor([50,255,50]).repeat([target_occ_coords.shape[0], 1])
                    target_points = torch.cat([target_occ_coords, target_colors], dim=1)
                    target_points[:, :3] *= self.voxel_size

                    target_colors_blue = torch.FloatTensor([255,50,50]).repeat([target_occ_coords.shape[0], 1])
                    target_points_blue = torch.cat([target_occ_coords, target_colors_blue], dim=1)
                    target_points_blue[:, :3] *= self.voxel_size
                    # print("target size: ", target_points.shape)

                    # points = torch.cat([input_points, pred_points, target_points], dim=0)
                    input_points = input_points
                    inout_points = torch.cat([input_points, target_points], dim=0)
                    inout_points_blue = torch.cat([input_points, target_points_blue], dim=0)
                    inpred_points = torch.cat([input_points, pred_points], dim=0)

                    #
                    msg1 = PointCloud2()
                    msg1.header.frame_id = "map"
                    msg1.height = 1
                    msg1.width = input_points.shape[0]
                    msg1.fields = [
                        PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('r', 12, PointField.FLOAT32, 1),
                        PointField('g', 16, PointField.FLOAT32, 1),
                        PointField('b', 20, PointField.FLOAT32, 1)
                        ]
                    msg1.is_bigendian = False
                    msg1.point_step = 24 #12
                    msg1.row_step = 24 * msg1.width
                    msg1.is_dense = False
                    msg1.data = input_points.reshape([-1]).cpu().numpy().tostring()
                    #
                    msg2 = deepcopy(msg1)
                    msg2.width = inout_points.shape[0]
                    msg2.row_step = 24 * msg2.width
                    msg2.data = inout_points.reshape([-1]).cpu().numpy().tostring()

                    #
                    msg0 = deepcopy(msg1)
                    msg0.width = inout_points_blue.shape[0]
                    msg0.row_step = 24 * msg2.width
                    msg0.data = inout_points_blue.reshape([-1]).cpu().numpy().tostring()

                    #
                    msg3 = deepcopy(msg1)
                    msg3.width = inpred_points.shape[0]
                    msg3.row_step = 24 * msg2.width
                    msg3.data = inpred_points.reshape([-1]).cpu().numpy().tostring()
                    
                    print("post process time: %fs"%(time.time() - start_time))

                    self.output_pub.publish(msg0)
                    # rospy.sleep(2)
                    raw_input('press key to continue')

                    self.output_pub.publish(msg1)
                    # rospy.sleep(2)
                    raw_input('press key to continue')
                    self.output_pub.publish(msg2)
                    # rospy.sleep(2)
                    raw_input('press key to continue')

                    self.output_pub.publish(msg3)
                    # rospy.sleep(2)                    
                    raw_input('press key to continue')
                    # self.output_pub.publish(msg0)
                    # # rospy.sleep(2)
                    # raw_input('press key to continue')                    
                    # self.output_pub.publish(msg1)
                    # # rospy.sleep(1)
                    # raw_input('press key to continue')
                    # self.output_pub.publish(msg2)
                    # # rospy.sleep(1)
                    # raw_input('press key to continue')
                    # self.output_pub.publish(msg3)
                    # raw_input('press key to continue')
                    # # rospy.sleep(2)
                    # if rospy.is_shutdown():
                    #     exit()

    def init_size(self, dimx, dimy, dimz):
        self.pointers = torch.zeros((dimx, dimy, dimz, 3)).float()
        for x in range(self.pointers.shape[0]):
            self.pointers[x, :, :, 0] = x
        for y in range(self.pointers.shape[1]):
            self.pointers[:, y, :, 1] = y
        for z in range(self.pointers.shape[2]):
            self.pointers[:, :, z, 2] = z

if __name__ == '__main__':
    worker = ocp_net()
    worker.vis()