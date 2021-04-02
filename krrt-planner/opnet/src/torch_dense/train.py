from __future__ import division
from __future__ import print_function

import argparse
import os, sys, time
import shutil
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter
import gc

import transform
import data_util
import scene_dataloader
import model_dense
import model_dense_simple
import loss as loss_util
from IPython import embed

# retrain: h2, no flip
# python2 train.py --gpu 0 --retrain ./logs/h3_64_NF2/model-iter56000-epoch55.pth --data_path /home/wlz/h3_64_629_denses --train_file_list ../filelists/h3_64_629_denses_train.txt --val_file_list ../filelists/h3_64_629_denses_val.txt --save logs/h3_64_NF3 --max_epoch 8 --num_hierarchy_levels 2 --save_epoch 1 --batch_size 20 --weight_missing_geo 2 --decay_lr 2
# train: h2, flip
# python2 train.py --g 0 --data_path /media/wlz/SSD/h3_80_size10_106 --train_file_list ../filelists/h3_64_629_denses_train_small.txt --val_file_list ../filelists/h3_64_629_denses_val_small.txt --save logs/TEST --max_epoch 25 --save_epoch 3 --batch_size 20 --num_hierarchy_levels 2 --weight_missing_geo 2 --decay_lr 7 --flip
# python2 train.py --g 0  --data_path /media/wlz/SSD/h3_80_size10_106 --train_file_list ../filelists/10cm_64_16_train.txt --val_file_list ../filelists/10cm_64_16_val.txt --save logs/10cm_sdf_occ --max_epoch 30 --save_epoch 3 --batch_size 20 --num_hierarchy_levels 2 --weight_missing_geo 3 --decay_lr 10 --flip
# python2 train.py --retrain ./logs/5cm__occ-occ+sdf-simple-aspp/model-epoch-10.pth --data_path /media/wlz/SSD/h3_64_629_denses --train_file_list ../filelists/h3_64_629_denses_train.txt --val_file_list ../filelists/h3_64_629_denses_val.txt --save logs/5cm__occ-occ+sdf-simple-aspp --max_epoch 30 --save_epoch 3 --batch_size 20 --num_hierarchy_levels 2 --weight_missing_geo 3 --decay_lr 20 --flip
# python2 train.py --data_path /media/wlz/SSD/h3_80_40 --train_file_list ../filelists/5cm_80_40_train.txt --val_file_list ../filelists/5cm_80_40_val.txt --save logs/NEW5cm-occ+sdf-simple-weight --max_epoch 30 --save_epoch 3 --batch_size 20 --num_hierarchy_levels 2 --weight_missing_geo 3 --decay_lr 10 --flip

# train: h2, no flip, occ
# python2 train.py --g 0 --data_path /media/wlz/h3_64_629_denses --train_file_list ../filelists/h3_64_629_denses_train.txt --val_file_list ../filelists/h3_64_629_denses_val.txt --save logs/h2_occ1_NF --max_epoch 20 --save_epoch 3 --batch_size 20 --num_hierarchy_levels 2 --weight_missing_geo 2 --decay_lr 5
# python2 train.py --g 0 --retrain ./logs/5cm__occ-occ/model-epoch-2.pth --data_path /media/wlz/h3_64_629_denses --train_file_list ../filelists/h3_64_629_denses_train.txt --val_file_list ../filelists/h3_64_629_denses_val.txt --save logs/h2_occ_NF --max_epoch 50 --save_epoch 3 --batch_size 20 --num_hierarchy_levels 2 --weight_missing_geo 1 --scheduler_step_size 3

WRITER = SummaryWriter(comment="-" + "new-5cm-occ+sdf-simple-weight5")

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data_path', required=True, help='path to data')
parser.add_argument('--val_data_path', default='', help='path to validation data')
parser.add_argument('--train_file_list', required=True, help='path to file list of train data')
parser.add_argument('--val_file_list', default='', help='path to file list of val data')
parser.add_argument('--save', default='./logs', help='folder to output model checkpoints')
# model params
parser.add_argument('--bool_input', dest='bool_input', action='store_true')
parser.add_argument('--flip', dest='flipped', action='store_true')
parser.add_argument('--retrain', type=str, default='', help='model to load from')
parser.add_argument('--input_dim', type=int, default=0, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
parser.add_argument('--no_logweight_target_sdf', dest='logweight_target_sdf', action='store_false')
# train params
parser.add_argument('--num_hierarchy_levels', type=int, default=3, help='#hierarchy levels (must be > 1).')
parser.add_argument('--num_iters_per_level', type=int, default=2000, help='#iters before fading in training for next level.')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--max_epoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--decay_lr', type=int, default=10, help='decay learning rate by half every n epochs')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay.')
parser.add_argument('--weight_sdf_loss', type=float, default=1.0, help='weight sdf loss vs occ.')
parser.add_argument('--weight_missing_geo', type=float, default=3.0, help='weight missing geometry vs rest of sdf.')
parser.add_argument('--vis_dfs', type=int, default=0, help='use df (iso 1) to visualize')
parser.add_argument('--use_loss_masking', dest='use_loss_masking', action='store_true')
parser.add_argument('--no_loss_masking', dest='use_loss_masking', action='store_false')
parser.add_argument('--scheduler_step_size', type=int, default=0, help='#epochs before scheduler step (0 for each epoch)')

parser.set_defaults(no_pass_occ=False, no_pass_feats=False, logweight_target_sdf=False, use_loss_masking=True)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.weight_missing_geo >= 1)
assert( args.num_hierarchy_levels > 1 )
if args.input_dim == 0: # set default values
    args.input_dim = 2 ** (3+args.num_hierarchy_levels)
    #TODO FIX THIS PART
    if '64' in args.data_path:
        args.input_dim = (64, 64, 48)
    elif '32' in args.data_path:
        args.input_dim = (32, 32, 48)
    if '96' in args.data_path:
        args.input_dim = (96, 96, 48)

args.input_nf = 1 # 2
UP_AXIS = 0
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

# create model
model = model_dense_simple.GenModel(args.encoder_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.retrain:
    print('loading model:', args.retrain)
    checkpoint = torch.load(args.retrain)
    args.start_epoch = args.start_epoch if args.start_epoch != 0 else checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict']) #, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])

last_epoch = -1 if not args.retrain else args.start_epoch - 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr, gamma=0.1, last_epoch=last_epoch)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.decay_lr // 2, eta_min=1e-6, last_epoch=last_epoch)
# data files
train_files, val_files = data_util.get_train_files(args.data_path, args.train_file_list, args.val_file_list, args.val_data_path)
_OVERFIT = False
if len(train_files) == 1:
    _OVERFIT = True
    args.use_loss_masking = False
num_overfit_train = 0 if not _OVERFIT else 640
num_overfit_val = 0 if not _OVERFIT else 160
print('#train files = ', len(train_files))
print('#val files = ', len(val_files))
train_dataset = scene_dataloader.DenseSceneDataset(train_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0, num_overfit_train, flipped=args.flipped, 
                                                    trans=transform.MyTransforms([transform.AddGaussianNoise(),
                                                                                #transform.AddPepperNoise(0.98, p=0.3), 
                                                                                transform.AddRandomFlip()], 
                                                                                random_lift=transform.RandomliftFloor(), max_lift=1)
                                                )
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=scene_dataloader.collate_dense, drop_last=True) # collate_fn=scene_dataloader.collate
if len(val_files) > 0:
    val_dataset = scene_dataloader.DenseSceneDataset(val_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0, num_overfit_val, flipped=args.flipped,
                                                    trans=transform.MyTransforms([#transform.AddGaussianNoise(), 
                                                            transform.AddRandomFlip()], 
                                                        )
                                                )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=scene_dataloader.collate_dense, drop_last=True) # collate_fn=scene_dataloader.collate

_SPLITTER = ','


def print_log_info(epoch, iter, mean_train_losses, mean_train_l1pred, mean_train_l1tgt, mean_train_ious, mean_val_losses, mean_val_l1pred, mean_val_l1tgt, mean_val_ious, time, log):
    return
    splitters = ['Epoch: ', ' iter: '] if log is None else ['', ',']
    values = [epoch, iter]
    values.extend(mean_train_losses)
    for h in range(len(mean_train_losses)):
        id = 'total' if h == 0 else str(h-1)
        id = 'sdf' if h + 1 == len(mean_train_losses) else id
        if log is None:
            splitters.append(' loss_train(' + id + '): ')
        else:
            splitters.append(',')
    # values.extend([mean_train_l1pred, mean_train_l1tgt])
    # if log is None:
    #     splitters.extend([' train_l1pred: ', ' train_l1tgt: '])
    # else:
    #     splitters.extend([',', ','])
    values.extend(mean_train_ious)
    for h in range(len(mean_train_ious)):
        id = str(h)
        if log is None:
            splitters.append(' iou_train(' + id + '): ')
        else:
            splitters.append(',')
    if mean_val_losses is not None:
        values.extend(mean_val_losses)
        for h in range(len(mean_val_losses)):
            id = 'total' if h == 0 else str(h-1)
            id = 'sdf' if h + 1 == len(mean_val_losses) else id
            if log is None:
                splitters.append(' loss_val(' + id + '): ')
            else:
                splitters.append(',')
        # values.extend([mean_val_l1pred, mean_val_l1tgt])
        # if log is None:
        #     splitters.extend([' val_l1pred: ', ' val_l1tgt: '])
        # else:
        #     splitters.extend([',', ','])
        values.extend(mean_val_ious)
        for h in range(len(mean_val_ious)):
            id = str(h)
            if log is None:
                splitters.append(' iou_val(' + id + '): ')
            else:
                splitters.append(',')
    else:
        splitters.extend([''] * (len(mean_train_losses) + len(mean_train_ious) + 2))
        values.extend([''] * (len(mean_train_losses) + len(mean_train_ious) + 2))
    values.append(time)
    if log is None:
        splitters.append(' time: ')
    else:
        splitters.append(',')
    info = ''
    for k in range(len(splitters)):
        if log is None and isinstance(values[k], float):
           info += splitters[k] + '{:.6f}'.format(values[k])
        else:
           info += splitters[k] + str(values[k])
    if log is None:
        print(info, file=sys.stdout)
    else:
        print(info, file=log)

def print_log(log, epoch, iter, train_losses, train_l1preds, train_l1tgts, train_ious, val_losses, val_l1preds, val_l1tgts, val_ious, time):
    return
    train_losses = np.array(train_losses)
    # train_l1preds = np.array(train_l1preds)
    # train_l1tgts = np.array(train_l1tgts)
    train_ious = np.array(train_ious)
    mean_train_losses = [(-1 if np.all(x < 0) else np.mean(x[x >= 0])) for x in train_losses]
    # mean_train_l1pred = -1 if (len(train_l1preds) == 0 or np.all(train_l1preds < 0)) else np.mean(train_l1preds[train_l1preds >= 0])
    # mean_train_l1tgt = -1 if (len(train_l1tgts) == 0 or np.all(train_l1tgts < 0))  else np.mean(train_l1tgts[train_l1tgts >= 0])
    mean_train_ious = [(-1 if np.all(x < 0) else np.mean(x[x >= 0])) for x in train_ious]
    mean_val_losses = None
    # mean_val_l1pred = None
    # mean_val_l1tgt = None
    mean_val_ious = None
    if val_losses:
        val_losses = np.array(val_losses)
        # val_l1preds = np.array(val_l1preds)
        # val_l1tgts = np.array(val_l1tgts)
        val_ious = np.array(val_ious)
        mean_val_losses = [-1 if np.all(x < 0) else np.mean(x[x >= 0]) for x in val_losses]
        # mean_val_l1pred = -1 if (len(val_l1preds) == 0 or np.all(val_l1preds < 0))  else np.mean(val_l1preds[val_l1preds >= 0])
        # mean_val_l1tgt = -1 if (len(val_l1tgts) == 0 or np.all(val_l1tgts < 0))  else np.mean(val_l1tgts[val_l1tgts >= 0])
        mean_val_ious = [-1 if np.all(x < 0) else np.mean(x[x >= 0]) for x in val_ious]
        print_log_info(epoch, iter, mean_train_losses, 0, 0, mean_train_ious, mean_val_losses, 0, 0, mean_val_ious, time, None)
        print_log_info(epoch, iter, mean_train_losses, 0, 0, mean_train_ious, mean_val_losses, 0, 0, mean_val_ious, time, log)
    else:
        print_log_info(epoch, iter, mean_train_losses, 0, 0, mean_train_ious, None, None, None, None, time, None)
        print_log_info(epoch, iter, mean_train_losses, 0, 0, mean_train_ious, None, None, None, None, time, log)
    log.flush()

def get_loss_weights(iter, num_hierarchy_levels, num_iters_per_level, factor_l1_loss):

    weights = np.zeros(num_hierarchy_levels+1, dtype=np.float32)
    cur_level = iter // num_iters_per_level
    if cur_level > num_hierarchy_levels:
        weights.fill(1)
        weights[-1] = factor_l1_loss
        if iter == (num_hierarchy_levels + 1) * num_iters_per_level:
            print('[iter %d] updating loss weights:' % iter, weights)
        return weights
    for level in range(0, cur_level+1):
        weights[level] = 1.0
    step_factor = 20
    fade_amount = max(1.0, min(100, num_iters_per_level//step_factor))
    fade_level = iter % num_iters_per_level
    cur_weight = 0.0
    l1_weight = 0.0
    if fade_level >= num_iters_per_level - fade_amount + step_factor:
        fade_level_step = (fade_level - num_iters_per_level + fade_amount) // step_factor
        cur_weight = float(fade_level_step) / float(fade_amount//step_factor)
    if cur_level+1 < num_hierarchy_levels:
        weights[cur_level+1] = cur_weight
    elif cur_level < num_hierarchy_levels:
        l1_weight = factor_l1_loss * cur_weight
    else:
        l1_weight = 1.0
    weights[-1] = l1_weight
    if iter % num_iters_per_level == 0 or (fade_level >= num_iters_per_level - fade_amount + step_factor and (fade_level - num_iters_per_level + fade_amount) % step_factor == 0):
        print('[iter %d] updating loss weights:' % iter, weights)

    # for testing
    # weights = np.ones(num_hierarchy_levels+1, dtype=np.float32)
    
    return weights

def train(epoch, iter, dataloader, log_file, output_save):
    start_time = time.time()

    train_losses = [ [] for i in range(args.num_hierarchy_levels+2) ]
    train_bceloss = []
    train_sdfloss = []
    train_precision = [] 
    train_recall = []
    train_f1score = []

    model.train()
    start = time.time()
    
    num_batches = len(dataloader)
    for t, sample in enumerate(dataloader):
        loss_weights = get_loss_weights(iter, args.num_hierarchy_levels, args.num_iters_per_level, args.weight_sdf_loss)
        if epoch == args.start_epoch and t == 0:
            print('[iter %d/epoch %d] loss_weights' % (iter, epoch), loss_weights)

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

        optimizer.zero_grad()
        output_sdf, output_occs = model(inputs, loss_weights)        
        # loss, losses = loss_util.compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs[0], args.use_loss_masking, known)
        bce_loss, sdf_loss, losses, last_loss = loss_util.compute_loss_dense(output_sdf, output_occs, target_for_sdf, target_for_occs, 
            target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs,
             args.use_loss_masking, known, flipped=args.flipped)
        loss = bce_loss * 10 + sdf_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        train_losses[0].append(loss.item())
        if last_loss > 0:
            train_bceloss.append(last_loss)
        train_sdfloss.append(bce_loss.item())

        output_visual = (output_save and t + 1 == num_batches and output_sdf is not None)
        compute_pred_occs = (iter % 20 == 0)  or output_visual

        # if compute_pred_occs:
        pred_occs = [None] * args.num_hierarchy_levels
        vis_occs = [None] * args.num_hierarchy_levels
        for h in range(args.num_hierarchy_levels):
            factor = 2**(args.num_hierarchy_levels-h-1)
            if isinstance(output_occs[h], list):
                continue
            # output_occs[h][1] = torch.nn.Sigmoid()(output_occs[h][1][:,0].detach()) > 0.5
            output_occs[h] = output_occs[h].detach()
            vis_occs[h] = output_occs[h][:, 1, :, :, :].unsqueeze(1)

        for h in range(args.num_hierarchy_levels):
            train_losses[h+1].append(losses[h])

        # if len(output_occs[-1]) is not 0:
        if output_sdf is not None:
            output_sdf.detach()
            # occ
            pred_occ = output_sdf[:,0].squeeze()
            ## sdf
            # pred_occ = pred_occ.abs() < (args.truncation / 2)
            # occ
            pred_occ = pred_occ > 0.1

            target_occ = target_for_occs[-1].squeeze()
            input_occ = inputs.detach().squeeze()
            precision, recall, f1score = loss_util.compute_dense_occ_accuracy(input_occ, target_occ, pred_occ, truncation=args.truncation)
        else:
            precision, recall, f1score = 0,0,0

        train_precision.append(precision)
        train_recall.append(recall)
        train_f1score.append(f1score)

        train_losses[args.num_hierarchy_levels+1].append(losses[-1])

        if output_sdf is not None:
            output_sdf = output_sdf.detach()

        iter += 1
        WRITER.add_scalar("loss/train", loss.item(), iter)

        if iter % 20 == 0:
            took = time.time() - start
            print_log(log_file, epoch, iter, train_losses, None, None, train_precision, None, None, None, None, took)
        if iter % 2000 == 0:
            torch.save({'epoch': epoch,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch)))

    if args.scheduler_step_size == 0:
        scheduler.step()
    else:
        args.scheduler_step_size -= 1

    WRITER.add_scalar("epoch_loss/train/total", np.mean(train_losses[0]), epoch)
    if len(train_bceloss) > 0:
        WRITER.add_scalar("epoch_loss/train/bce", np.mean(train_bceloss), epoch)
    WRITER.add_scalar("epoch_loss/train/precision", np.mean(train_precision), epoch)
    WRITER.add_scalar("epoch_loss/train/recall", np.mean(train_recall), epoch)
    WRITER.add_scalar("epoch_loss/train/f1", np.mean(train_f1score), epoch)
    
    print(" test epoch {} used {} seconds".format(epoch, time.time() - start_time))
    
    return train_losses, train_precision, train_recall, train_f1score, iter, loss_weights

def test(epoch, iter, loss_weights, dataloader, log_file, output_save):
    start_time = time.time()
    val_losses = [ [] for i in range(args.num_hierarchy_levels+2) ]
    val_bceloss = []
    val_sdfloss = []
    val_precision = []
    val_recall = []
    val_f1score = []

    val_ious = [ [] for i in range(args.num_hierarchy_levels) ]
    model.eval()

    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            sdfs = sample['sdf']
            if sdfs.shape[0] < args.batch_size:
                continue  # maintain same batch size
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

            output_sdf, output_occs = model(inputs, loss_weights)        
            # loss, losses = loss_util.compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs[0], args.use_loss_masking, known)
            bce_loss, sdf_loss, losses, last_loss = loss_util.compute_loss_dense(output_sdf, output_occs, target_for_sdf, target_for_occs, 
                target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs,
                args.use_loss_masking, known, flipped=args.flipped)
            loss = bce_loss * 10 + sdf_loss

            val_losses[0].append(loss.item())
            if last_loss > 0:
                val_bceloss.append(last_loss)
            WRITER.add_scalar("loss/val", loss.item(), iter + t)
            output_visual = (output_save and t + 1 == num_batches and output_sdf is not None)
            compute_pred_occs = (t % 20 == 0) or output_visual

            # if compute_pred_occs:
            pred_occs = [None] * args.num_hierarchy_levels
            vis_occs = [None] * args.num_hierarchy_levels
            for h in range(args.num_hierarchy_levels):
                factor = 2**(args.num_hierarchy_levels-h-1)
                if isinstance(output_occs[h], list):
                    continue
                # output_occs[h][1] = torch.nn.Sigmoid()(output_occs[h][1][:,0].detach()) > 0.5
                output_occs[h] = output_occs[h].detach()
                vis_occs[h] = output_occs[h][:, 1, :, :, :].unsqueeze(1)

            for h in range(args.num_hierarchy_levels):
                val_losses[h+1].append(losses[h])

            # if len(output_occs[-1]) is not 0:
            if output_sdf is not None:
                pred_occ = output_sdf[:,0].squeeze()
                # pred_occ = pred_occ.abs() < (args.truncation / 2)
                pred_occ = pred_occ > 0.5
                target_occ = target_for_occs[-1].squeeze()
                input_occ = inputs.squeeze()
                precision, recall, f1score = loss_util.compute_dense_occ_accuracy(input_occ, target_occ, pred_occ, truncation=args.truncation)
            else:
                precision, recall, f1score = 0,0,0

            val_precision.append(precision)
            val_recall.append(recall)
            val_f1score.append(f1score)
            val_losses[args.num_hierarchy_levels+1].append(losses[-1])

            if output_visual:
                data_util.save_dense_predictions(os.path.join(args.save, 'iter%d-epoch%d' % (iter, epoch), 'val'), sample['name'],
                     input_occ.cpu().numpy(), target_occ.cpu().numpy(), pred_occ.cpu().numpy(), args.truncation, flipped=args.flipped)

    WRITER.add_scalar("epoch_loss/val/total", np.mean(val_losses[0]), epoch)
    if len(val_bceloss) > 0:
        WRITER.add_scalar("epoch_loss/val/bce", np.mean(val_bceloss), epoch)

    WRITER.add_scalar("epoch_loss/val/precision", np.mean(val_precision), epoch)
    WRITER.add_scalar("epoch_loss/val/recall", np.mean(val_recall), epoch)
    WRITER.add_scalar("epoch_loss/val/f1", np.mean(val_f1score), epoch)

    print(" test epoch {} used {} seconds".format(epoch, time.time() - start_time))
    return val_losses, val_precision, val_recall, val_f1score,


def main():
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    elif not _OVERFIT:
        raw_input('warning: save dir %s exists, press key to delete and continue' % args.save)

    data_util.dump_args_txt(args, os.path.join(args.save, 'args.txt'))
    log_file = open(os.path.join(args.save, 'log.csv'), 'w')
    headers = ['epoch','iter','train_loss(total)']
    for h in range(args.num_hierarchy_levels):
        headers.append('train_loss(' + str(h) + ')')
    headers.extend(['train_loss(sdf)', 'train_l1-pred', 'train_l1-tgt'])
    for h in range(args.num_hierarchy_levels):
        headers.append('train_iou(' + str(h) + ')')
    headers.extend(['time'])
    log_file.write(_SPLITTER.join(headers) + '\n')
    log_file.flush()

    has_val = len(val_files) > 0
    log_file_val = None
    if has_val:
        headers = headers[:-1]
        headers.append('val_loss(total)')
        for h in range(args.num_hierarchy_levels):
            headers.append('val_loss(' + str(h) + ')')
        headers.extend(['val_loss(sdf)', 'val_l1-pred', 'val_l1-tgt'])
        for h in range(args.num_hierarchy_levels):
            headers.append('val_iou(' + str(h) + ')')
        headers.extend(['time'])
        log_file_val = open(os.path.join(args.save, 'log_val.csv'), 'w')
        log_file_val.write(_SPLITTER.join(headers) + '\n')
        log_file_val.flush()
    # start training
    print('starting training...')
    iter = args.start_epoch * (len(train_dataset) // args.batch_size) * 2
    for epoch in range(args.start_epoch, args.max_epoch):
        start = time.time()

        train_losses, train_l1preds, train_l1tgts, train_ious, iter, loss_weights = train(epoch, iter, train_dataloader, log_file, output_save=(epoch % args.save_epoch == 0))
        if has_val:
            val_losses, val_l1preds, val_l1tgts, val_ious = test(epoch, iter, loss_weights, val_dataloader, log_file_val, output_save=(epoch % args.save_epoch == 0) & (epoch != 0))

        took = time.time() - start
        if has_val:
            print_log(log_file_val, epoch, iter, train_losses, train_l1preds, train_l1tgts, train_ious, val_losses, None, None, val_ious, took)
        else:
            print_log(log_file, epoch, iter, train_losses, train_l1preds, train_l1tgts, train_ious, None, None, None, None, took)
        torch.save({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-epoch-%s.pth' % epoch))
    log_file.close()
    if has_val:
        log_file_val.close()



if __name__ == '__main__':
    try:
        main()
        WRITER.close()
    except KeyboardInterrupt:
        WRITER.close()
        exit()

