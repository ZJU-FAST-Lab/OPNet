#!/usr/bin/env python2

import argparse
import data_util
from IPython import embed

import argparse

import numpy as np
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda
import rospy
import rospkg
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from occ_grid.srv import PredictPCL, PredictPCLResponse

import trt_runner
# import torch

# import gc
import time

# import model_dense

# python2 net_node_trt.py --model_path /home/wlz/vox_ws/src/opnet/models/h2-flip-14.pth --input_topic /voxblox_node/ctsdf_pointcloud --flipped --num_hierarchy_levels 2 --dimx 64 --dimz 48
# python2 net_node_trt.py --onnx_path ./h2_nf_80.onnx --engine_path ./h2_nf_80.trt

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--onnx_path', required=True, help='path to onnx model')
parser.add_argument('--engine_path', required=True, help='path to trt eigen')

parser.add_argument('--output', default='./output', help='folder to output predictions')
# test params
# parser.add_argument('--max_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--truncation', type=float, default=2, help='truncation in voxels')
parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel_size in meters')
parser.add_argument('--flipped', dest='flipped', action='store_true')
parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.add_argument('--dimx', type=int, default=80, help='boundingbox dim x/y')
parser.add_argument('--dimz', type=int, default=48, help='boundingbox dim z')
parser.add_argument('--occ_thresord', type=float, default=1.0, help='thredsord value of occ and freespace')

# ros params
parser.add_argument('--service_name', type=str, default="/occ_map/pred", help='ros service name')

# parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
args = parser.parse_args()
# assert( not (args.no_pass_feats or args.no_pass_occ) )
# assert( args.num_hierarchy_levels > 1 )
# args.input_nf = 1
print(args)


# todo: compare np and torch+cuda
POINTER = np.zeros([args.dimx, args.dimx, args.dimz, 3], dtype=np.float32, order='C')
for x in range(POINTER.shape[0]):
    POINTER[x, :, :, 0] = x
for y in range(POINTER.shape[1]):
    POINTER[:, y, :, 1] = y
for z in range(POINTER.shape[2]):
    POINTER[:, :, z, 2] = z

SUM = args.dimx * args.dimx * args.dimz




ENGINE = None
INPUTS = OUTPUTS = BINDINGS = STREAM = CONTEXT = None

RUNNER = trt_runner.trt_runner(args.onnx_path, args.engine_path)

class trt_net(object):

    def __init__(self):
        """ Constructor """
        self.init_params()

        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_runner = trt_runner.trt_runner(
            args.onnx_path, args.engine_path, args.dimx, args.dimx, args.dimz)

        # self.test()

    def __del__(self):
        """ Destructor """

        self.cuda_ctx.pop()
        del self.trt_runner
        del self.cuda_ctx

    def clean_up(self):
        """ Release cuda memory """
        if self.trt_runner is not None:
            self.cuda_ctx.pop()
            del self.trt_runner
            del self.cuda_ctx

    def init_params(self):
        """ Initializes ros parameters """
        
        # rospack = rospkg.RosPack()
        # package_path = rospack.get_path("yolov4_trt_ros")
        self.dimx = rospy.get_param("/dimx", "80")
        self.dimy = rospy.get_param("/dimx", "80")
        self.dimz = rospy.get_param("/dimx", "48")
    #     self.category_num = rospy.get_param("/category_number", 80)
    #     self.conf_th = rospy.get_param("/confidence_threshold", 0.5)
    #     self.show_img = rospy.get_param("/show_image", True)
        # self.pcl_sub = rospy.Subscriber(
        #     self.pcl_topic, CenteredPointcloud, self.tsdf_callback, queue_size=1) # , buff_size=1920*1080*3 ?

        # self.occ_pub = rospy.Publisher(args.output_topic, PointCloud2, queue_size=1)
        # self.added_occ_pub = rospy.Publisher(args.output_topic + "_added", PointCloud2, queue_size=1)
        self.service= rospy.Service(args.service_name, PredictPCL, self.pred_server)



    def test(self):
        dummy_input = np.random.randn(1, args.dimx, args.dimx, args.dimz).astype(np.float32)
        self.trt_runner.inference(dummy_input)

    def tsdf_callback(self, data):

        # self.test()
        # print("hhahahahha")

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
        input = input.astype(np.float32)
        # print("shape after s to d", input.shape)

        input[input > args.truncation] = args.truncation
        input[input < -args.truncation] = -args.truncation

        input.reshape([1, args.dimx, args.dimx, args.dimz])

        # input = data_util.tsdf_to_bool(input, trunc=2.0)

        # for grids whose tsdf value > -2, we consider it as already known
        known_mask = (input > -2).squeeze() # (80,80,48)
        original_occ = (np.abs(input) < args.occ_thresord).squeeze()
        original_free = (input > args.occ_thresord).squeeze()
        original_unkown = (input < - 2).squeeze()
        # print("INPUT: %f  occ, %f  free, %f  unkwn"%(original_occ.float().sum()/SUM, original_free.float().sum()/SUM, original_unkown.float().sum()/SUM))

        time2 = time.time()
        print("prepocess time: %fs"%(time2 - time1))
    
        output_occ = self.trt_runner.inference(input)

        time3 = time.time()
        print("model time: %fs"%(time3 - time2))

        output_occ = np.abs(output_occ) < args.occ_thresord
            # fix conflicts with input in known grids
        # embed()
        output_occ[known_mask] = original_occ[known_mask]
        # added_occ = output_occ.float() > original_occ.float()
        added_occ = output_occ > original_occ

        if output_occ is not None:

            # added_occ = added_occ.squeeze()
            # todo: compare of np and torch+cuda
            occ_coords_1 = POINTER[added_occ].reshape([-1, 3])
            num_points = occ_coords_1.shape[0]
            # added_color_1 = torch.ones(num_points).unsqueeze(1) * 10
            added_color_1 = np.ones([num_points, 1], dtype=np.float32) * 10

            # occ_coords_1 = torch.cat((occ_coords_1, added_color_1), 1)
            occ_coords_1 = np.concatenate((occ_coords_1, added_color_1), 1)

            # occ_coords = torch.cat((occ_coords, occ_coords_1), 0) 
            occ_coords = occ_coords_1
            print("added size: ",occ_coords.shape, added_occ.shape)    

            if occ_coords.shape[0] > 0:
                occ_coords *= args.voxel_size
                occ_coords[:, 0] += offset_x
                occ_coords[:, 1] += offset_y
                occ_coords[:, 2] += offset_z
                # occ_coords = occ_coords.cpu().numpy()

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
                print("msg: ", occ_coords.shape, len(msg.data), msg.row_step)
                # print("postpocess time: %fs"%(time.time() - time3))
                self.added_occ_pub.publish(msg)
            
            #publish added occ
            # original_unkown = original_input <= -2

            # original_unkown = original_unkown.squeeze()
            # original_occ = original_occ.squeeze()
            occ_coords = POINTER[original_occ].reshape([-1, 3])
            print("known occ size: ",occ_coords.shape)

            if occ_coords.shape[0] > 0:
                occ_coords *= args.voxel_size
                occ_coords[:, 0] += offset_x
                occ_coords[:, 1] += offset_y
                occ_coords[:, 2] += offset_z
                # occ_coords = occ_coords.cpu().numpy()

                msg2 = data.points
                msg2.height = 1
                msg2.width = occ_coords.shape[0]
                msg2.fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),]
                msg2.is_bigendian = False
                msg2.point_step = 12 #12
                msg2.row_step = 12 * occ_coords.shape[0]
                msg2.is_dense = int(np.isfinite(occ_coords).all())
                msg2.data = occ_coords.tostring() #np.asarray(occ_coords, np.float32).tostring()
                print("msg2: ", occ_coords.shape, len(msg2.data), msg2.row_step)

                self.occ_pub.publish(msg2)        

        time4 = time.time()
        print("post pocess time: %fs"%(time4 - time3))    

        return

    def pred_server(self, req):
        time1 = time.time()
        dimx = req.dim_x
        dimy = req.dim_y
        dimz = req.dim_z
        input = req.input
        input = np.array(req.input).astype(np.float32)
        input = input.reshape([1, dimx, dimy, dimz])

        # input = data_util.tsdf_to_bool(input, trunc=2.0)

        # for grids whose tsdf value > -2, we consider it as already known
        print((input > 0).shape)
        known_mask = (input > 0).squeeze(0) # (80,80,48)
        original_occ = (input > 0).squeeze(0)
        # original_free = (input > args.occ_thresord).squeeze()
        # original_unkown = (input < - 2).squeeze()
        time2 = time.time()
        print("prepocess time: %fs"%(time2 - time1))
    
        output_occ = self.trt_runner.inference(input)

        time3 = time.time()
        print("model time: %fs"%(time3 - time2))
        output_occ = np.abs(output_occ) < args.occ_thresord
            # fix conflicts with input in known grids
        # embed()
        output_occ[known_mask] = original_occ[known_mask]
        output_occ = output_occ.astype(np.float32).reshape([-1]).tolist()
        # print("sdf_to_occ: ", sdf_to_occ.shape, sdf_to_occ.sum())

        # wlz: original occ:input, sdf_to_occ:output, added_occ:added
        # added_occ = sdf_to_occ.float() > original_occ.float()
        # output_occ = output_occ.tolist()
        print("post process time: %fs"%(time.time() - time3))

        return PredictPCLResponse(output_occ)

def main():
    # global INPUTS, OUTPUTS, BINDINGS, STREAM, CONTEXT
    # with get_engine(args.onnx_path, args.engine_path) as engine, engine.create_execution_context() as context:
    #     # ENGINE = engine
    #     print("GET ENGINE SUCCEED")
    #     ENGINE = engine
    #     CONTEXT = context
    #     INPUTS, OUTPUTS, BINDINGS, STREAM = common.allocate_buffers(engine)
    my_net = trt_net()
    rospy.init_node('pcl_predictor', anonymous=True)
    # rospy.Subscriber(args.input_topic, CenteredPointcloud, tsdf_callback)

    # spin() simply keeps python from exiting until this node is stopped

    # dummy_input = np.random.randn(1, args.dimx, args.dimx, args.dimz).astype(np.float32)
    # RUNNER.inference(dummy_input, dummy_input.shape[1:])
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.on_shutdown(my_net.clean_up())
        print("Shutting down")

if __name__ == '__main__':
    main()
