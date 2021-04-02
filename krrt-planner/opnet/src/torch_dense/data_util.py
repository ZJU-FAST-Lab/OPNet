import os, sys, struct
import scipy.io as sio
import numpy as np
import torch
import random
import math
import plyfile
from IPython import embed

# import marching_cubes.marching_cubes as mc


def get_train_files(data_path, file_list, val_file_list, val_data_path=None):
    if not val_data_path:
        val_data_path = data_path
    names = open(file_list).read().splitlines()
    if not '.' in names[0]:
        names = [name + '__0__.sdf' for name in names]
    files = [os.path.join(data_path, f) for f in names]
    # print("TRAIN FILE: ")
    # for file in files:
    #     print(file)
    val_files = []
    if val_file_list:
        val_names = open(val_file_list).read().splitlines()
        val_files = [os.path.join(val_data_path, f) for f in val_names]
    return files, val_files


def dump_args_txt(args, output_file):
    with open(output_file, 'w') as f:
        f.write('%s\n' % str(args))


# locs: hierarchy, then batches
def compute_batchids(output_occs, output_sdf, batch_size):
    batchids = [None] * (len(output_occs) + 1)
    for h in range(len(output_occs)):
        batchids[h] = [None] * batch_size
        for b in range(batch_size):
            batchids[h][b] = output_occs[h][0][:,-1] == b
    batchids[-1] = [None] * batch_size
    for b in range(batch_size):
        batchids[-1][b] = output_sdf[0][:,-1] == b
    return batchids


# locs: zyx ordering
def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    # print(dimz, dimx, dimy, nf_values, values.dtype)
    # dense = np.zeros([dimx, dimy, dimz], dtype=values.dtype)
    dense = np.zeros([dimx, dimy, dimz], dtype=np.float32)
    dense.fill(default_val)
    # print('dense', dense.shape)
    # print('locs', locs.shape, locs[:10, :3])
    # print('values', values.shape)
    dense[locs[:,0], locs[:,1], locs[:,2]] = values
    if nf_values > 1:
        return dense.reshape([dimx, dimy, dimz, nf_values])
    return dense.reshape([dimx, dimy, dimz])

def sparse_to_dense_np_flipped(locs, values, dimx, dimy, dimz, default_val=0):
    assert default_val == 0
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    # print(dimz, dimx, dimy, nf_values, values.dtype)
    dense = np.zeros([dimx, dimy, dimz, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    # print('dense', dense.shape)
    # print('locs', locs.shape, locs[:10, :3])
    # print('values', values.shape)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense.reshape([dimx, dimy, dimz, nf_values])
    return dense.reshape([dimx, dimy, dimz,])

def dense_to_sparse_np(grid, thresh):
    locs = np.where(np.abs(grid) < thresh)
    values = grid[locs[0], locs[1], locs[2]]
    locs = np.stack(locs)
    return locs, values


def load_train_file(file, hie=4):
    # print("FILE NAME: ", file)
    fin = open(file, 'rb')
    #'Q': ctype = unsigned long long, calsize = 8
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    # print("TOTAL DIM: ", dimx, dimy, dimz)
    # 'f': ctype = float, calsize = 4
    voxelsize = struct.unpack('f', fin.read(4))[0]
    # print("Voxel size: ", voxelsize)
    # tf matrix
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # print("world2grid: ", world2grid)
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    # print("input num: ", num)
    # 'I': ctype = unsigned int, calsize = 4 
    # num of points * 3 (xyz location)
    input_locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    # input_locs = np.flip(input_locs,1).copy() # convert to zyx ordering
    # print("input_locs: ", input_locs[:10])
    input_sdfs = struct.unpack('f'*num, fin.read(num*4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    # print("input_sdfs", input_sdfs.shape, input_sdfs.max(), input_sdfs[:10])
    input_sdfs /= voxelsize
    # target data
    num = struct.unpack('Q', fin.read(8))[0]
    # print("target num: ", num)
    target_locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    target_locs = np.asarray(target_locs, dtype=np.int32).reshape([num, 3])
    # target_locs = np.flip(target_locs,1).copy() # convert to zyx ordering
    target_sdfs = struct.unpack('f'*num, fin.read(num*4))
    target_sdfs = np.asarray(target_sdfs, dtype=np.float32)
    target_sdfs /= voxelsize
    # print("target_sdfs", target_sdfs.shape, target_sdfs.max(), target_sdfs[:10])
    # embed()
    target_sdfs = sparse_to_dense_np(target_locs, target_sdfs[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
    # known data
    num = struct.unpack('Q', fin.read(8))[0]
    # print("known num: ", num)
    assert(num == dimx * dimy * dimz)
    # 'B': ctype = unsigned char, calsize = 1
    target_known = struct.unpack('B'*dimz*dimy*dimx, fin.read(dimz*dimy*dimx))
    target_known = np.asarray(target_known, dtype=np.uint8).reshape([dimx, dimy, dimz])
    # print("target_known", target_known.shape, target_known.max())
    # pre-computed hierarchy
    hierarchy = []
    factor = 2
    for h in range(hie - 1):
        # print("DIM: ,", h, dimx//factor, dimy//factor, dimz//factor)
        num = struct.unpack('Q', fin.read(8))[0]
        # print("hie%d num: "%h, num)
        hlocs = struct.unpack('I'*num*3, fin.read(num*3*4))
        hlocs = np.asarray(hlocs, dtype=np.int32).reshape([num, 3])
        # hlocs = np.flip(hlocs,1).copy() # convert to zyx ordering
        hvals = struct.unpack('f'*num, fin.read(num*4))
        hvals = np.asarray(hvals, dtype=np.float32)
        hvals /= voxelsize
        # print("hvals%d"%h, hvals.shape, hvals.max(), np.sum(np.abs(hvals)<=3), hvals[:10])
        grid = sparse_to_dense_np(hlocs, hvals[:,np.newaxis], dimx//factor, dimy//factor, dimz//factor, -float('inf'))
        hierarchy.append(grid)
        factor *= 2
    hierarchy.reverse()
    # print("hierarchy: ", len(hierarchy))
    return [input_locs, input_sdfs], target_sdfs, [dimx, dimy, dimz], world2grid, target_known, hierarchy
    
def load_dense_train_file(file, hie=4):
    # print("FILE NAME: ", file)
    fin = open(file, 'rb')
    #'Q': ctype = unsigned long long, calsize = 8
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    # print("TOTAL DIM: ", dimx, dimy, dimz)
    # 'f': ctype = float, calsize = 4
    voxelsize = struct.unpack('f', fin.read(4))[0]
    # print("Voxel size: ", voxelsize)
    # tf matrix
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # print("world2grid: ", world2grid)
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    assert(num == dimx * dimy * dimz)
    # print("input num: ", num)
    # 'I': ctype = unsigned int, calsize = 4 
    # num of points * 3 (xyz location)
    input_sdfs = struct.unpack('f'*num, fin.read(num*4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32).reshape([dimx, dimy, dimz])
    # print("input_sdfs", input_sdfs.shape, input_sdfs.max(), input_sdfs[:10])
    input_sdfs /= voxelsize
    # target data
    num = struct.unpack('Q', fin.read(8))[0]
    assert(num == dimx * dimy * dimz)
    # print("target num: ", num)
    target_sdfs = struct.unpack('f'*num, fin.read(num*4))
    target_sdfs = np.asarray(target_sdfs, dtype=np.float32).reshape([dimx, dimy, dimz])
    target_sdfs /= voxelsize
    # print("target_sdfs", target_sdfs.shape, target_sdfs.max(), target_sdfs[:10])
    # known data
    num = struct.unpack('Q', fin.read(8))[0]
    assert(num == dimx * dimy * dimz)
    # 'B': ctype = unsigned char, calsize = 1
    target_known = struct.unpack('B'*dimz*dimy*dimx, fin.read(dimz*dimy*dimx))
    target_known = np.asarray(target_known, dtype=np.uint8).reshape([dimx, dimy, dimz])
    hierarchy = []
    for h in range(hie - 1):
        num = struct.unpack('Q', fin.read(8))[0]
        hvals = struct.unpack('f'*num, fin.read(num*4))
        hvals = np.asarray(hvals, dtype=np.float32).reshape([dimx//(pow(2,h+1)), dimy//(pow(2,h+1)), dimz//(pow(2,h+1))])
        hvals /= voxelsize
        hierarchy.append(hvals)
    hierarchy.reverse()

    return input_sdfs, target_sdfs, [dimx, dimy, dimz], world2grid, target_known, hierarchy
       

def load_scene(file):
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    # print("dim: ", dimx, dimy, dimz)
    voxelsize = struct.unpack('f', fin.read(4))[0]
    # print("voxelsize: ", voxelsize)
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    world2grid *= 50
    world2grid[3][3] = 1
    # data 
    num = struct.unpack('Q', fin.read(8))[0]
    # print("num: ", num)
    locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
    # print("locs: ", locs[:10])
    # print("locs: ", locs.shape)
    # locs = np.flip(locs,1).copy() # convert to zyx ordering
    sdf = struct.unpack('f'*num, fin.read(num*4))
    sdf = np.asarray(sdf, dtype=np.float32)
    # print("sdf: ", sdf[:20])
    sdf /= voxelsize
    
    fin.close()
    return [locs, sdf], [dimx, dimy, dimz], world2grid


def load_scene_known(file):
    #assert os.path.isfile(file)
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    known = struct.unpack('B'*dimz*dimy*dimx, fin.read(dimz*dimy*dimx))
    known = np.asarray(known, dtype=np.uint8).reshape([dimz, dimy, dimx])
    fin.close()
    return known


def preprocess_sdf_np(sdf, truncation):
    sdf[sdf < -truncation] = -truncation
    sdf[sdf > truncation] = truncation
    return sdf
def preprocess_sdf_pt(sdf, truncation, flipped=False): 
    if not flipped:
        sdf[sdf < -truncation] = -truncation
        sdf[sdf > truncation] = truncation
    # # wlz: flipped sdf
    else:
        sdf[sdf < -truncation] = -0
        sdf[sdf > truncation] = 0    
        sdf = torch.sign(sdf) * (truncation - sdf.abs())
    return sdf
def unprocess_sdf_pt(sdf, truncation):
    return sdf

def dense_to_points(input):
    # collect verts from sdf
    verts = []
    for x in range(input.shape[-3]):
        for y in range(input.shape[-2]):
            for z in range(input.shape[-1]):
                val = input[x, y, z]
                if val > 0:
                    verts.append(np.array([x, y, z, val]))  # center of voxel
                    verts[-1][:3] += 0.5
    if len(verts) == 0:
        return None
    verts = np.stack(verts)
    return verts

def tsdf_to_bool(input, trunc=3.0):
    # known_mask = input > -2
    # # occ
    # occ_layer = torch.zeros_like(input)
    # occ_layer[input.abs() < trunc] = 1
    # # free
    # occ_layer[input >= trunc] = - 1

    # # unknown
    # known_layer = (~ known_mask).float()

    # return torch.cat((occ_layer.unsqueeze_(0), known_layer.unsqueeze_(0)), 0)

    # occ
    occ_layer = np.zeros_like(input)
    occ_mask = (input < trunc) & (input > -1) # [-1 - trunc] is considered as occ 
    occ_layer[occ_mask] = 1 # - torch.max(input[occ_mask], 1) / trunc
    # free
    occ_layer[input < 0] = - 1

    return occ_layer

def visualize_sdf_as_points(sdf, iso, output_file, transform=None):
    # collect verts from sdf
    sdf = sdf.squeeze()
    verts = []
    for x in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for z in range(sdf.shape[2]):
                if abs(sdf[x,y,z]) < iso:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts)
    visualize_points(verts, output_file, transform)

def visualize_fsdf_as_points(sdf, iso, output_file, transform=None, thresh=1):
    # collect verts from sdf
    sdf = sdf.squeeze()
    verts = []
    for x in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for z in range(sdf.shape[2]):
                if abs(sdf[x,y,z]) > thresh:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts)
    visualize_points(verts, output_file, transform)

def visualize_sparse_sdf_as_points(sdf_locs, sdf_vals, iso, output_file, transform=None):
    # collect verts from sdf
    mask = np.abs(sdf_vals) < iso
    verts = sdf_locs[:,:3][mask]
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:,::-1] + 0.5
    visualize_points(verts, output_file, transform)

def visualize_occ_as_points(occ, thresh, output_file, transform=None, thresh_max = float('inf')):
    # collect verts from sdf
    occ = occ.squeeze()
    verts = []
    for x in range(occ.shape[0]):
        for y in range(occ.shape[1]):
            for z in range(occ.shape[2]):
                val = float(occ[x, y, z])
                if val > thresh and val < thresh_max:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    #print('[visualize_occ_as_points]', output_file, len(verts))
    verts = np.stack(verts)
    visualize_points(verts, output_file, transform)

def visualize_sparse_locs_as_points(locs, output_file, transform=None):
    # collect verts from sdf
    verts = locs[:,:3]
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    #print('[visualize_occ_as_points]', output_file, len(verts))
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:,::-1] + 0.5
    visualize_points(verts, output_file, transform)

def visualize_sparse_flocs_as_points(occ, output_file, transform=None, thresh=0):
    # collect verts from sdf
    locs = occ[0]
    feats = occ[1]
    mask = feats[:, 0] > thresh
    verts = locs[:,:3][mask]
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    #print('[visualize_occ_as_points]', output_file, len(verts))
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:,::-1] + 0.5
    visualize_points(verts, output_file, transform)

def visualize_points(points, output_file, transform=None, colors=False):
    verts = points
    if verts is None:
        return
    if verts.shape[1] == 3 and colors:
        raise ValueError('VISUALIZE: NO COLOR INFO')
    if transform is not None:
        x = np.ones((verts.shape[0], 4))
        x[:, :3] = verts[:, :3]
        x = np.matmul(transform, np.transpose(x))
        x = np.transpose(x)
        verts = np.divide(x[:, :3], x[:, 3, None])

    ext = os.path.splitext(output_file)[1]
    # if colors is not None:
    #     colors = np.clip(colors, 0, 1)
    # if colors is not None or ext == '.obj':
    #     output_file = os.path.splitext(output_file)[0] + '.obj'
    #     num_verts = len(verts)
    #     with open(output_file, 'w') as f:
    #         for i in range(num_verts):
    #             v = verts[i]
    #             if colors is None:
    #                 f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #             else:
    #                 f.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], colors[i,0], colors[i,1], colors[i,2]))
    # elif ext == '.ply':
    if not colors:
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = plyfile.PlyElement.describe(verts,'vertex')
        plyfile.PlyData([el]).write(output_file)
    else:
        output_file = os.path.splitext(output_file)[0] + '.xyzrgb'
        num_verts = len(verts)
        with open(output_file, 'w') as f:
            for i in range(num_verts):
                v = verts[i]
                color = v[-1]
                if color == 1:
                    color = [1,0,0]
                elif color == 2:
                    color = [0,1,0]
                elif color ==3:
                    color = [0,0,1]
                f.write('%f %f %f %f %f %f\n' % (v[0], v[1], v[2], color[0],color[1],color[2]))    

def visiualize_diff(output_file, mask1, mask2, trunc=3.0, flipped=False):
    # ''' 
# visiualize sdf1, highlight(sdf1 - sdf2)
#     output: 0-none, 1-r, 2-g, 3-b
# '''
    mask1 = mask1.squeeze()
    mask2 = mask2.squeeze()
    
    output = torch.zeros(mask2.shape)
    # if flipped:
    #     mask2 = np.abs(sdf2) > 0.5
    #     mask1 = np.abs(sdf1) > 0.5
    # else:
    #     mask2 = np.abs(sdf2) < trunc
    #     mask1 = np.abs(sdf1) < trunc

    output[mask1 > mask2] = 1
    output[mask1 < mask2] = 2
    output[mask1 & mask2] = 3
    verts = dense_to_points(output)
    visualize_points(verts, output_file, colors=True)

def make_scale_transform(scale):
    if isinstance(scale, int) or isinstance(scale, float):
        scale = [scale, scale, scale]
    assert( len(scale) == 3 )
    transform = np.eye(4, 4)
    for k in range(3):
        transform[k,k] = scale[k]
    return transform

def save_dense_predictions(output_path, names, inputs, target_for_occs, output_occs, truncation, thresh=1, flipped = False):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    trunc = truncation - 1.0
    ext = '.ply'

    for k in range(len(names)):
        name = names[k]
        # diff pred-input
        if output_occs is not None:
            mask_input = inputs[k] > 0.5 # input is occ
            # mask_pred = (output_sdf[k,0] < trunc) & (output_sdf[k,0] > -1)   
            mask_pred = output_occs[k] > 0.5
            mask_input_known = inputs[k] >= 0  
            # fix known region by input
            mask_pred[mask_input_known] = mask_input[mask_input_known]

            visiualize_diff(os.path.join(output_path, name + 'pred-input' + '.xyzrgb'),
                mask_pred, mask_input, trunc=trunc, flipped=flipped)
            if target_for_occs is not None:
                mask_target = target_for_occs[k] > 0.5
                visiualize_diff(os.path.join(output_path, name + 'target-pred' + '.xyzrgb'), 
                    mask_target, mask_pred, trunc=trunc, flipped=flipped)

