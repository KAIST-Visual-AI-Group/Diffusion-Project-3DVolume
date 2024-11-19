import multiprocessing as mp
import os
import sys
from itertools import product

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


def voxel_to_pointcloud(voxel_grid: torch.Tensor, vox_res=(128, 128, 128)):
    vox_res = np.array(vox_res)
    voxel_indices = torch.argwhere(voxel_grid > 0)
    normalized_pts = voxel_indices / (vox_res - 1)
    normalized_pts = normalized_pts.reshape(-1, 3)

    return normalized_pts


def distChamfer(x, y):
    kdtree1 = cKDTree(x)
    kdtree2 = cKDTree(y)

    dist1, _ = kdtree1.query(y)
    dist2, _ = kdtree2.query(x)

    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    # print("bye")
    return chamfer_dist


def processChamfer(xy):
    return distChamfer(xy[0], xy[1])


def pairwise_CD(sample_pcs, ref_pcs, use_multiprocessing=True):
    """
    In case you encounter the "OSError: too many open errors" error,
    you can increase the limit for open files using the `ulimit` command in Ubuntu.
    """
    num_sample = len(sample_pcs)
    num_ref = len(ref_pcs)
    total_num = num_sample * num_ref

    # [[0,0], [0,1], [0,num_ref-1], [1,0], ..., [num_sample-1,num_ref-1]]
    indices = [list(item) for item in product(range(num_sample), range(num_ref))]
    indices = torch.from_numpy(np.array(indices)).long()

    cd_results = []
    if use_multiprocessing:
        args = []
        for si, ri in indices:
            args.append((sample_pcs[si], ref_pcs[ri]))
        with tqdm(total=total_num) as pbar:
            pool = mp.Pool(mp.cpu_count())

            for result in pool.imap(processChamfer, args):
                cd_results.append(result)
                pbar.update()
            pool.close()
    else:
        pbar = tqdm(indices)
        for si, ri in pbar:
            sample_pc = sample_pcs[si]
            ref_pc = ref_pcs[ri]

            cd = distChamfer(sample_pc, ref_pc)
            cd_results.append(cd)

    cd_results = torch.tensor(cd_results).view(num_sample, num_ref)
    return cd_results


def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between distributions p and q.
    Args:
        p (torch.Tensor): Distribution p.
        q (torch.Tensor): Distribution q.
    Returns:
        torch.Tensor: KL divergence value.
    """
    # To avoid division by zero or log(0), add a small epsilon value
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return torch.sum(p * torch.log(p / q))


def jensen_shannon_divergence(voxel_set1, voxel_set2):
    """
    Compute the Jensen-Shannon Divergence between two 3D voxel sets.
    Args:
        voxel_set1 (torch.Tensor): Binary 3D voxel grid (shape: N1 x 128 x 128 x 128).
        voxel_set2 (torch.Tensor): Binary 3D voxel grid (shape: N2 x 128 x 128 x 128).
        N1 and N2: # of samples
    Returns:
        float: Jensen-Shannon Divergence value.
    """
    # Count the number of points lying within each voxel grid across all voxel samples.
    voxel_set1 = voxel_set1.sum(0)
    voxel_set2 = voxel_set2.sum(0)

    # Normalize the voxel grids to make them probability distributions
    voxel_set1_prob = voxel_set1 / voxel_set1.sum()
    voxel_set2_prob = voxel_set2 / voxel_set2.sum()

    # Compute the average distribution (M)
    mean_distribution = (voxel_set1_prob + voxel_set2_prob) / 2

    # Compute KLD(P1 || M) and KLD(P2 || M)
    kl1 = kl_divergence(voxel_set1_prob, mean_distribution)
    kl2 = kl_divergence(voxel_set2_prob, mean_distribution)

    # Jensen-Shannon Divergence is the mean of the two KL divergences
    jsd = (kl1 + kl2) / 2
    return jsd.item()


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    """
    Cut off matrices so that they become a square matrix.
    """
    num_x = Mxx.shape[0]
    num_y = Myy.shape[0]
    min_len = min(num_x, num_y)
    Mxx = Mxx[:min_len, :min_len]
    Mxy = Mxy[:min_len, :min_len]
    Myy = Myy[:min_len, :min_len]
    print(f"Mxx: {Mxx.shape} Mxy: {Mxy.shape} Myy: {Myy.shape}")
    # ========================

    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat(
        (torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0
    )
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False
    )

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "MMD": mmd.item(),
        "COV": cov.item(),
    }


if __name__ == "__main__":
    category = sys.argv[1]
    sample_path = sys.argv[2]
    
    assert category in ["chair", "airplane", "table"], f"{category} should be one of `chair`, `airplane`, or `table`."

    X_gen = torch.from_numpy(np.load(sample_path)).float()
    # X_gen = torch.rand(1000, 128, 128, 128)  # For testing.
    # X_gen = (X_gen > 0.98).float()

    shapenet_dir = "./data/hdf5_data"  # TODO: CHANGE THE PATH IF NEEDED.
    test_set_path = os.path.join(shapenet_dir, f"{category}_voxels_test.npy")
    val_set_path = os.path.join(shapenet_dir, f"{category}_voxels_val.npy")
    assert os.path.exists(test_set_path), f"{test_set_path} not exist."
    assert os.path.exists(val_set_path), f"{val_set_path} not exist."

    test_set = torch.from_numpy(np.load(test_set_path))
    val_set = torch.from_numpy(np.load(val_set_path))
    X_ref = torch.cat([test_set, val_set]).float()

    print("[*] Computing JSD...")
    jsd_score = jensen_shannon_divergence(X_gen, X_ref)
    print(f"JSD: {jsd_score}")

    print("[*] Computing MMD and COV...")
    sample_pcs = [voxel_to_pointcloud(x).numpy() for x in X_gen]
    ref_pcs = [voxel_to_pointcloud(x).numpy() for x in X_ref]

    M_sr = pairwise_CD(sample_pcs, ref_pcs)
    res_dic = lgan_mmd_cov(M_sr)
    for k, v in res_dic.items():
        print(f"{k}: {v}")
