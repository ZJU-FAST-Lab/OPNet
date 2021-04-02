# import torch
# import torchvision
# from test_model import GenModel

# dummy_input = torch.randn(1, 80, 80, 48, device='cuda')
# model = GenModel().cuda()

# # Providing input and output names sets the display names for values
# # within the model's graph. Setting these does not change the semantics
# # of the graph; it is only for readability.
# #
# # The inputs to the network consist of the flat list of inputs (i.e.
# # the values you would pass to the forward() method) followed by the
# # flat list of parameters. You can partially specify names, i.e. provide
# # a list here shorter than the number of inputs to the model, and we will
# # only set that subset of names, starting from the beginning.
# input_names = [ "input1" ]
# output_names = [ "output1" ]

# torch.onnx.export(model, dummy_input, "alexnet.onnx", keep_initializers_as_inputs=True, verbose=True, input_names=input_names, output_names=output_names)


#########################################


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
# import model_dense_h2
# from test_model_simple import GenModel
from test_model_nosdf import GenModel


import loss as loss_util

# python2 test_scene.py --gpu 0 --model_path ./logs/h3_64/model-iter14000-epoch35.pth --input_data_path ../data/data616  --test_file_list ../filelists/data616_train.txt --output output/h2_64f --max_to_vis 10 --num_hierarchy_levels 3
# python2 test_scene.py --gpu 0 --model_path ./logs/h2_64/model-epoch-5.pth --input_data_path /media/wlz/Elements/datagen/GenerateScans/output/h3_64  --test_file_list ../filelists/h3_64_train.txt --output output/mp/h3_64_train --max_to_vis 5
# python2 model_dense_script.py --gpu 0 --model_path /home/wlz/vox_ws/src/opnet/models/h2_NF_noise.pth 

# params
parser = argparse.ArgumentParser()
# data paths

# model params
# parser.add_argument('--flip', dest='flipped', action='store_true')
# parser.add_argument('--num_hierarchy_levels', type=int, default=2, help='#hierarchy levels.')
# parser.add_argument('--max_input_height', type=int, default=128, help='max height in voxels')
# parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
# parser.add_argument('--input_dim', type=int, default=128, help='voxel dim.')
# parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
# parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
# parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
# parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
# parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
# parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
# parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
# # test params
# parser.add_argument('--max_to_vis', type=int, default=10, help='max num to vis')
# parser.add_argument('--cpu', dest='cpu', action='store_true')


# parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
# args = parser.parse_args()
# assert( not (args.no_pass_feats and args.no_pass_occ) )
# assert( args.num_hierarchy_levels > 1 )
# args.input_nf = 1
# print(args)

# create model
# model = model_dense_h2.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense)
model = GenModel(8, 1, 16, 16, 1, True, True, True, True)
# if not args.cpu:
#     model = model.cuda()

dummy_input = torch.randn(1, 80, 120, 40, device='cuda')
model_path = '../../models/nosdf_v0.pth' 

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.to("cuda")
model.eval()
output = model(dummy_input)
print("output.shape: ", output.shape)


print('loaded model:', model_path)
# sm = torch.jit.script(model)
input_names = [ "input1" ]
output_names = [ "output1" ]
torch.onnx.export(model, dummy_input, "../../models/nosdf_80_120_40.onnx", keep_initializers_as_inputs=True, verbose=True, 
                input_names=input_names, output_names=output_names, opset_version=10)

print("save succeed")

