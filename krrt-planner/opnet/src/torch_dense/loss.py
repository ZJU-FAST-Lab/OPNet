
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import sparseconvnet as scn

import data_util
from IPython import embed

# UNK_THRESH = 2
UNK_THRESH = 3

UNK_ID = -1

def compute_targets(target, hierarchy, num_hierarchy_levels, truncation, use_loss_masking, known, flipped=False):
    '''INPUT: sdf & target sdf + hierarchy, known mask, ...
        OUTPUT: TARGET_SDF, TARGET_OCC(dense)
        OCC: abs(sdf) < truncation
    '''
    assert(len(target.shape) == 5)
    target_for_occs = [None] * num_hierarchy_levels
    target_for_hier = [None] * num_hierarchy_levels
    # done in scene_dataloader
    known_mask = None
    target_for_hier[-1] = target.clone()
    # occ: 1 occupied, 0 free, -1 unknown
    # wlz: flipped
    # if flipped:
    #     target_occ = (torch.abs(target) > 0).float()
    #     # target_occ = target_occ * torch.abs(target_for_sdf)  * 1.5 / truncation
    # else:
    target_occ = (torch.abs(target) <= 1).float()

    if use_loss_masking:
        # target_occ[known >= UNK_THRESH] = UNK_ID
        target_occ[target < -1] = UNK_ID
    target_for_occs[-1] = target_occ

    # factor = 2
    for h in range(num_hierarchy_levels-2,-1,-1):
        target_for_occs[h] = torch.nn.MaxPool3d(kernel_size=2, stride=2)(target_for_occs[h+1])
        target_for_hier[h] = hierarchy[h]

    # if use bool_input ## output sdf now
    # if flipped:
    #     target = data_util.tsdf_to_bool(target, trunc=truncation)
    #     for item in target_for_hier:
    #         item = data_util.tsdf_to_bool(item, trunc=truncation)

    return target, target_for_occs, target_for_hier

def compute_weights_missing_geo_dense(weight_missing_geo, input_occ, target_for_occs, target_for_sdf, truncation, flipped=False):
    '''
        punish geo loss of input (regions included in input which disapeared in prediction)
    '''
    num_hierarchy_levels = len(target_for_occs)
    weights = [None] * num_hierarchy_levels
    weights[-1] = torch.ones(target_for_occs[-1].shape).float().cuda()
    missing_mask_all = (input_occ < 0 ) & (target_for_occs[-1] >= 0)
    missing_mask_occ = (input_occ < 0 ) & (target_for_occs[-1] > 0)
    missing_mask_close = missing_mask_all & (target_for_sdf < truncation)
    # bonus occ
    weights[-1][missing_mask_occ] += (weight_missing_geo - 1)

    factor = 2
    for h in range(num_hierarchy_levels-2,-1,-1):
        weights[h] = torch.nn.MaxPool3d(kernel_size=2, stride=2)(weights[h+1])
        factor *= 2
    # print("loss weights: ", weights[-1].shape)
    return weights


def apply_log_transform(sdf):
    # return  sign * log(abs(sdf) + 1)
    sgn = torch.sign(sdf)
    out = torch.log(torch.abs(sdf) + 1)
    out = sgn * out
    return out


def compute_bce_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, weights, use_loss_masking, truncation=3, batched=True):
    ''' calculate loss of pred and target
        only count for target locs where is known
        bce with logits loss
    '''
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    # print("sparse_pred_locs: ", sparse_pred_locs[:,3].max(), sparse_pred_locs[:,2].max())

    predvalues = sparse_pred_vals.view(-1)
    flatlocs = sparse_pred_locs[:,3]*dims[0]*dims[1]*dims[2] + sparse_pred_locs[:,0]*dims[1]*dims[2] + sparse_pred_locs[:,1]*dims[2] + sparse_pred_locs[:,2]
    tgtvalues = dense_tgts.view(-1)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]
    if use_loss_masking:
        mask = tgtvalues != UNK_ID
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
        if weight is not None:
            weight = weight[mask]
    else:
        tgtvalues[tgtvalues == UNK_ID] = 0
    if batched:
        # wlz: flipped
        # loss = torch.abs(predvalues - tgtvalues)
        # loss = torch.mean(loss * weight)
        loss = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
        else:
            raise
    return loss

def compute_bce_dense(dense_inputs, dense_tgts, weights, use_loss_masking, truncation=3, batched=True, balance=False):
    ''' calculate loss of pred and target
        only count for target locs where is known
        bce with logits loss
    '''
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    assert(dense_inputs.shape == dense_tgts.shape)

    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    weight = weights

    # balance
    if balance:
        spatial_size = dense_inputs.shape[-1] * dense_inputs.shape[-2] * dense_inputs.shape[-3]
        occ_mask = dense_tgts > 0
        occ_num = occ_mask.float().sum()
        # occ / total
        occ_ratio = occ_num / ((dense_tgts != UNK_ID).float().sum() + 1)
        occ_bonus = (0.5 * (1 - occ_ratio)/ (1e-4 + occ_ratio)).clamp(0, 5)
        # print("loss-occratio: ", occ_ratio)
        weight[occ_mask] *= occ_bonus
    if use_loss_masking:
        mask = dense_tgts != UNK_ID
        tgtvalues = dense_tgts[mask]
        predvalues = dense_inputs[mask]
        if weight is not None:
            weight = weight[mask]
    else:
        tgtvalues = dense_tgts
        tgtvalues[tgtvalues == UNK_ID] = 0
    if batched:
        loss = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight, reduction='mean')
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight, reduction='mean')
        else:
            raise ValueError
    return loss

def compute_iou_sparse_dense(sparse_pred_locs, dense_tgts, use_loss_masking, truncation=3, batched=True): 
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    corr = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    union = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    for b in range(dense_tgts.shape[0]):
        tgt = dense_tgts[b,0]
        if sparse_pred_locs[b] is None:
            continue
        predlocs = sparse_pred_locs[b]
        # flatten locs # TODO not sure whats the most efficient way to compute this...
        predlocs = predlocs[:,0] * dims[1] * dims[2] + predlocs[:,1] * dims[2] + predlocs[:,2]
        tgtlocs = torch.nonzero(tgt == 1)
        tgtlocs = tgtlocs[:,0] * dims[1] * dims[2] + tgtlocs[:,1] * dims[2] + tgtlocs[:,2]
        if use_loss_masking:
            tgtlocs = tgtlocs.cpu().numpy()
            # mask out from pred
            mask = torch.nonzero(tgt == UNK_ID)
            mask = mask[:,0] * dims[1] * dims[2] + mask[:,1] * dims[2] + mask[:,2]
            predlocs = predlocs.cpu().numpy()
            if mask.shape[0] > 0:
                _, mask, _ = np.intersect1d(predlocs, mask.cpu().numpy(), return_indices=True)
                predlocs = np.delete(predlocs, mask)
        else:
            predlocs = predlocs.cpu().numpy()
            tgtlocs = tgtlocs.cpu().numpy()
        if batched:
            corr += len(np.intersect1d(predlocs, tgtlocs, assume_unique=True)) 
            union += len(np.union1d(predlocs, tgtlocs))
        else:
            corr[b] = len(np.intersect1d(predlocs, tgtlocs, assume_unique=True)) 
            union[b] = len(np.union1d(predlocs, tgtlocs))
    if not batched:
        return np.divide(corr, union)
    if union > 0:
        return corr/union
    return -1

def compute_dense_occ_accuracy(dense_inputs, dense_tgts, pred_occ, truncation=3): 
    '''
        dense_input: sdf
        dense_tgts: occ, 0/1/-1
        tp:true positive, fn: false nagetive
    '''
    if dense_inputs is None:
        return 0, 0, 0
    dense_inputs = dense_inputs.squeeze()
    dense_tgts = dense_tgts.squeeze()
    pred_occ = pred_occ.squeeze()
    assert (dense_inputs.shape == dense_tgts.shape and pred_occ.shape == dense_tgts.shape)
    # sdf
    # pred_occ = (pred_occ.abs() < truncation / 2)
    target_known_mask = (dense_tgts != UNK_ID)
    input_known_mask = (dense_inputs >= 0)
    input_lost_known = (target_known_mask > input_known_mask)

    dense_tgts = dense_tgts > 0.5
    pred_occ_in_lost = pred_occ & input_lost_known
    target_occ_in_lost = dense_tgts & input_lost_known

    success_pred_in_lost = pred_occ_in_lost & target_occ_in_lost

    recall = success_pred_in_lost.float().sum() / target_occ_in_lost .float().sum()
    precision = success_pred_in_lost.float().sum() / pred_occ_in_lost .float().sum()
    f1score = 2 * precision * recall / (precision + recall + 1e-4)

    return precision.cpu().numpy(), recall.cpu().numpy(), f1score.cpu().numpy()


def compute_l1_predsurf_dense(dense_inputs, dense_tgts, weights, use_log_transform, use_loss_masking, known, batched=True, 
    thresh=3, flipped=False):

    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    # assert(dense_inputs.shape == dense_tgts.shape)
    
    loss_func = nn.SmoothL1Loss(reduction='none')
    weight = weights
    tgtvalues = dense_tgts
    predvalues = dense_inputs.clamp(-thresh, thresh)
    if use_loss_masking:
        mask = known < UNK_THRESH
        predvalues = predvalues[mask]
        tgtvalues = tgtvalues[mask]
        if weight is not None:
            weight = weight[mask]
    # if use_log_transform and not flipped:
    predvalues = apply_log_transform(predvalues)
    tgtvalues = apply_log_transform(tgtvalues)
    if batched:
        loss = loss_func(predvalues, tgtvalues)
        if weight is not None:
            loss *= weight
    else:
        if dense_tgts.shape[0] == 1:
            if weight is not None:
                loss_ = torch.abs(predvalues - tgtvalues)
                loss[0] = torch.mean(loss_ * weight).item()
            else:
                loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise ValueError("batch size?")
    return loss.mean()


def occ_punish(occ, weight_mask, weight=1e-3):
    unknown_mask = weight_mask > 1.0
    occ = occ[unknown_mask]
    func = nn.Sigmoid()
    loss = func(occ).mean() * weight

    return loss

def sscnet_loss(output, target_sdf, target_occ, input_occ, weight_missing_geo=3.0, truncation=3.0):

    sdf_loss = 0
    output_occ = output[:, 0].unsqueeze(1)

    weights = compute_weights_missing_geo_dense(weight_missing_geo, input_occ, [target_occ], target_sdf, truncation, flipped=True)
    valid_mask = target_occ != UNK_ID

    pred_occ = output_occ[valid_mask]
    target_occ = target_occ[valid_mask]
    weight = weights[-1][valid_mask] 
    bce_loss = F.binary_cross_entropy_with_logits(pred_occ, target_occ, weight=weight, reduction='mean')
    # log

    if output.shape[1] == 2:
        output_sdf = output[:, 1].unsqueeze(1)
        pred_sdf = output_sdf[valid_mask]
        target_sdf = target_sdf[valid_mask]
        pred_sdf = apply_log_transform(pred_sdf)
        target_sdf = apply_log_transform(target_sdf)
        loss_func = nn.SmoothL1Loss(reduction='none')
        sdf_loss = loss_func(pred_sdf, target_sdf) * weight
        sdf_loss = sdf_loss.mean()

    true_lost = F.binary_cross_entropy_with_logits(pred_occ, target_occ, weight=torch.ones_like(weight), reduction='mean')
    return bce_loss, sdf_loss, true_lost

def compute_loss_dense(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, 
    truncation, use_log_transform=False, weight_missing_geo=1, inputs=None, use_loss_masking=True, known=None, batched=True, 
    flipped=False):
    # embed()
    # print(output_occs.shape)
    # print(target_for_occs.shape)
    occ_punishment = 0.0

    assert(len(output_occs) == len(target_for_occs))
    batch_size = target_for_sdf.shape[0]
    bce_loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    sdf_loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    losses = [] if batched else [[] for i in range(len(output_occs) + 1)]
    weights = [None] * len(target_for_occs)
    if weight_missing_geo > 1:
        weights = compute_weights_missing_geo_dense(weight_missing_geo, inputs, target_for_occs, target_for_sdf, truncation, flipped=flipped)
    # loss hierarchical
    for h in range(len(output_occs)):
        if isinstance(output_occs[h], list) or loss_weights[h] == 0:
            if batched:
                losses.append(-1)
            else:
                losses[h].extend([-1] * batch_size)
            continue
        cur_loss_occ = compute_bce_dense(output_occs[h][:,0].unsqueeze(1), target_for_occs[h], weights[h], use_loss_masking, 
                batched=batched, balance=False)
        cur_known = None if not use_loss_masking else (target_for_occs[h] == UNK_ID)*UNK_THRESH
        cur_loss_sdf = compute_l1_predsurf_dense(output_occs[h][:,1].unsqueeze(1), target_for_hier[h], weights[h], 
                use_log_transform, use_loss_masking, cur_known, batched=batched, thresh=truncation, flipped=flipped)
        # cur_loss = cur_loss_occ + cur_loss_sdf
        if batched:
            bce_loss += loss_weights[h] * cur_loss_occ
            sdf_loss += loss_weights[h] * cur_loss_sdf
            losses.append(cur_loss_occ.item() + cur_loss_sdf.item() * 0.3)
        else:
            raise ValueError("no batch no train")
            # loss += loss_weights[h] * cur_loss
            # losses[h].extend(cur_loss)
    # loss sdf
    # if output_sdf is not None and loss_weights[-1] > 0:
    #     cur_loss = compute_l1_predsurf_dense(output_sdf, target_for_sdf, weights[-1], use_log_transform, use_loss_masking, 
    #         known, batched=batched, thresh=truncation, flipped=flipped)
    #     if batched:
    #         sdf_loss += loss_weights[-1] * cur_loss
    #         losses.append(cur_loss.item())
    #     else:
    #         sdf_loss += loss_weights[-1] * cur_loss
    #         losses[len(output_occs)].extend(cur_loss)
    if loss_weights[-2] > 0: 
        bce_loss += loss_weights[-2] * occ_punish(output_occs[-1][:,0].unsqueeze(1), weights[-1], occ_punishment)
    last_loss = -1
    if output_sdf is not None and loss_weights[-1] > 0:
        if output_sdf.shape[1] == 2:
        # occ + sdf
            last_loss_sdf  = compute_l1_predsurf_dense(output_sdf[:,1].unsqueeze(1), target_for_sdf, weights[-1], use_log_transform, use_loss_masking, 
                known, batched=batched, thresh=truncation, flipped=flipped)
        last_loss_bce = compute_bce_dense(output_sdf[:,0].unsqueeze(1), target_for_occs[-1], weights[-1], use_loss_masking, 
            batched=batched, balance=False)
        if batched:
            bce_loss += loss_weights[-1] * last_loss_bce
            bce_loss += loss_weights[-1] * occ_punish(output_sdf[:,0].unsqueeze(1), weights[-1], occ_punishment)
            if output_sdf.shape[1] == 2:
                sdf_loss += loss_weights[-1] * last_loss_sdf
            losses.append(last_loss_bce.item())
            weight = (weights[-1] > 0).float()
            real_loss = compute_bce_dense(output_sdf[:,0].unsqueeze(1), target_for_occs[-1], weight, use_loss_masking, 
            batched=batched, balance=False)
            last_loss = real_loss.item()
        else:
            bce_loss += loss_weights[-1] * last_loss_bce
            losses[len(output_occs)].extend(last_loss_sdf)


    else:
        if batched:
            losses.append(-1)
        else:
            losses[len(output_occs)].extend([-1] * batch_size)
    return bce_loss, sdf_loss, losses, last_loss

def compute_loss_nosurf(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, 
    truncation, use_log_transform=False, weight_missing_geo=1, inputs=None, use_loss_masking=True, known=None, batched=True, 
    flipped=False):
    # embed()
    # print(output_occs.shape)
    # print(target_for_occs.shape)
    occ_punishment = 0.0

    assert(len(output_occs) == len(target_for_occs))
    batch_size = target_for_sdf.shape[0]
    bce_loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    sdf_loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    losses = [] if batched else [[] for i in range(len(output_occs) + 1)]
    weights = [None] * len(target_for_occs)
    if weight_missing_geo > 1:
        weights = compute_weights_missing_geo_dense(weight_missing_geo, inputs, target_for_occs, target_for_sdf, truncation, flipped=flipped)
    # loss hierarchical
    for h in range(len(output_occs) - 1):
        if isinstance(output_occs[h], list) or loss_weights[h] == 0:
            if batched:
                losses.append(-1)
            else:
                losses[h].extend([-1] * batch_size)
            continue
        cur_loss_occ = compute_bce_dense(output_occs[h][:,0].unsqueeze(1), target_for_occs[h], weights[h], use_loss_masking, 
                batched=batched, balance=False)
        cur_known = None if not use_loss_masking else (target_for_occs[h] == UNK_ID).float() *UNK_THRESH
        cur_loss_sdf = compute_l1_predsurf_dense(output_occs[h][:,1].unsqueeze(1), target_for_hier[h], weights[h], 
                use_log_transform, use_loss_masking, cur_known, batched=batched, thresh=truncation, flipped=flipped)
        # cur_loss = cur_loss_occ + cur_loss_sdf
        if batched:
            bce_loss += loss_weights[h] * cur_loss_occ
            sdf_loss += loss_weights[h] * cur_loss_sdf
            losses.append(cur_loss_occ.item() + cur_loss_sdf.item())
        else:
            raise ValueError("no batch no train")
            # loss += loss_weights[h] * cur_loss
            # losses[h].extend(cur_loss)

    real_loss = -1
    if isinstance(output_occs[-1], list) or loss_weights[-2] == 0:
        if batched:
            losses.append(-1)
        else:
            losses[h].extend([-1] * batch_size)
    else:
        cur_loss_occ = compute_bce_dense(output_occs[-1][:,0].unsqueeze(1), target_for_occs[-1], weights[-1], use_loss_masking, 
                batched=batched, balance=False)
        cur_known = None if not use_loss_masking else (target_for_occs[-1] == UNK_ID).float() * UNK_THRESH
        cur_loss_sdf = compute_l1_predsurf_dense(output_occs[-1][:,1].unsqueeze(1), target_for_hier[-1], weights[-1], 
                use_log_transform, use_loss_masking, cur_known, batched=batched, thresh=truncation, flipped=flipped)

        weight = (weights[-1] > 0).float()
        real_loss = compute_bce_dense(output_occs[-1][:,0].unsqueeze(1), target_for_occs[-1], weight, use_loss_masking, 
            batched=batched, balance=False).item()
        # cur_loss = cur_loss_occ + cur_loss_sdf
        if batched:
            bce_loss += loss_weights[-2] * cur_loss_occ
            sdf_loss += loss_weights[-2] * cur_loss_sdf
            losses.append(cur_loss_occ.item() + cur_loss_sdf.item())
        else:
            raise ValueError("no batch no train")

    return bce_loss, sdf_loss, losses, real_loss

def compute_loss_nosdf(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, 
    truncation, use_log_transform=False, weight_missing_geo=1, inputs=None, use_loss_masking=True, known=None, batched=True, 
    flipped=False):
    # embed()
    # print(output_occs.shape)
    # print(target_for_occs.shape)
    occ_punishment = 0.0

    assert(len(output_occs) == len(target_for_occs))
    batch_size = target_for_sdf.shape[0]
    bce_loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    sdf_loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    losses = [] if batched else [[] for i in range(len(output_occs) + 1)]
    weights = [None] * len(target_for_occs)
    if weight_missing_geo > 1:
        weights = compute_weights_missing_geo_dense(weight_missing_geo, inputs, target_for_occs, target_for_sdf, truncation, flipped=flipped)
    # loss hierarchical
    for h in range(len(output_occs) - 1):
        if isinstance(output_occs[h], list) or loss_weights[h] == 0:
            if batched:
                losses.append(-1)
            else:
                losses[h].extend([-1] * batch_size)
            continue
        cur_loss_occ = compute_bce_dense(output_occs[h][:,0].unsqueeze(1), target_for_occs[h], weights[h], use_loss_masking, 
                batched=batched, balance=False)
        cur_known = None if not use_loss_masking else (target_for_occs[h] == UNK_ID).float() *UNK_THRESH
        # cur_loss = cur_loss_occ + cur_loss_sdf
        if batched:
            bce_loss += loss_weights[h] * cur_loss_occ
            losses.append(cur_loss_occ.item())
        else:
            raise ValueError("no batch no train")
            # loss += loss_weights[h] * cur_loss
            # losses[h].extend(cur_loss)

    real_loss = -1
    if isinstance(output_occs[-1], list) or loss_weights[-2] == 0:
        if batched:
            losses.append(-1)
        else:
            losses[h].extend([-1] * batch_size)
    else:
        cur_loss_occ = compute_bce_dense(output_occs[-1][:,0].unsqueeze(1), target_for_occs[-1], weights[-1], use_loss_masking, 
                batched=batched, balance=False)
        cur_known = None if not use_loss_masking else (target_for_occs[-1] == UNK_ID).float() * UNK_THRESH

        weight = (weights[-1] > 0).float()
        real_loss = compute_bce_dense(output_occs[-1][:,0].unsqueeze(1), target_for_occs[-1], weight, use_loss_masking, 
            batched=batched, balance=False).item()
        # cur_loss = cur_loss_occ + cur_loss_sdf
        if batched:
            bce_loss += loss_weights[-2] * cur_loss_occ
            losses.append(cur_loss_occ.item())
        else:
            raise ValueError("no batch no train")

    return bce_loss, 0, losses, real_loss

def compute_l1_tgtsurf_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, truncation, use_loss_masking, known, 
    batched=True, thresh=None):
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    batch_size = dense_tgts.shape[0]
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    pred_dense = torch.ones(batch_size * dims[0] * dims[1] * dims[2]).to(dense_tgts.device)
    fill_val = -truncation
    pred_dense.fill_(fill_val)
    if thresh is not None:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) <= thresh)
    else:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) < truncation)
    batchids = tgtlocs[:,0]
    tgtlocs = tgtlocs[:,0]*dims[0]*dims[1]*dims[2] + tgtlocs[:,2]*dims[1]*dims[2] + tgtlocs[:,3]*dims[2] + tgtlocs[:,4]
    tgtvalues = dense_tgts.view(-1)[tgtlocs]
    flatlocs = sparse_pred_locs[:,3]*dims[0]*dims[1]*dims[2] + sparse_pred_locs[:,0]*dims[1]*dims[2] + sparse_pred_locs[:,1]*dims[2] + sparse_pred_locs[:,2]
    pred_dense[flatlocs] = sparse_pred_vals.view(-1)
    predvalues = pred_dense[tgtlocs]
    if use_loss_masking:
        mask = known < UNK_THRESH
        mask = mask.view(-1)[tgtlocs]
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
    if batched:
        loss = torch.mean(torch.abs(predvalues - tgtvalues)).item()
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise
    return loss

