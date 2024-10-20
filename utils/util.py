import math
import os

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.cluster import KMeans

class RandomSampler:
    def __init__(self, stack_n=1000, fast_sample=True):
        self.stack_n = stack_n
        self.pre_samples = []
        self.fast_sample = fast_sample

    def randn(self, n, k, d):
        if self.fast_sample and len(self.pre_samples) > self.stack_n:
            self.pre_samples = np.array(self.pre_samples)
            return self.pre_samples[np.random.choice(np.arange(self.stack_n), n)]
        randn_sample = []
        for _ in range(n):
            k_samples = KMeans(n_clusters=k).fit(np.random.normal(size=(1000, d)))
            randn_sample.append(k_samples.cluster_centers_ * 0.8)
        self.pre_samples.extend(randn_sample) if self.fast_sample else None
        return np.array(randn_sample)

random = RandomSampler()

def compute_batch_metric(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=1).min(dim=0)[0]
    FDEs = temp[:, -1, :].min(dim=0)[0]
    pred_best = pred[temp[:, -1, :].argmin(dim=0), :, range(pred.size(2)), :]
    pred_gt_stack = torch.stack([pred_best, gt.permute(1, 0, 2)], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    num_interp, thres = 4, 0.2
    pred_fp = pred[:, [0], :, :]
    pred_rel = pred[:, 1:] - pred[:, :-1]
    pred_rel_dense = pred_rel.div(num_interp).unsqueeze(dim=2).repeat_interleave(repeats=num_interp, dim=2).contiguous()
    pred_rel_dense = pred_rel_dense.reshape(pred.size(0), num_interp * (pred.size(1) - 1), pred.size(2), pred.size(3))
    pred_dense = torch.cat([pred_fp, pred_rel_dense], dim=1).cumsum(dim=1)
    col_mask = pred_dense[:, :3 * num_interp + 2].unsqueeze(dim=2).repeat_interleave(repeats=pred.size(2), dim=2)
    col_mask = (col_mask - col_mask.transpose(2, 3)).norm(p=2, dim=-1)
    col_mask = col_mask.add(torch.eye(n=pred.size(2), device=pred.device)[None, None, :, :]).min(dim=1)[0].lt(thres)
    COLs = col_mask.sum(dim=1).gt(0).type(pred.type()).mean(dim=0).mul(100)
    return ADEs, FDEs


def generate_statistics_matrices(V):
    r"""generate mean and covariance matrices from the network output."""

    mu = V[:, :, 0:2]
    sx = V[:, :, 2].exp()
    sy = V[:, :, 3].exp()
    corr = V[:, :, 4].tanh()

    cov = torch.zeros(V.size(0), V.size(1), 2, 2, device=V.device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy

    return mu, cov
