import os
import random
import re
import warnings


import numpy as np
import ot
from ot.utils import list_to_array, get_parameter_pair
from ot.backend import get_backend
import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from scipy import sparse
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt

from models import ODEFunc2


def to_np(data):
    return data.detach().cpu().numpy()


def generate_steps(groups):
    return list(zip(groups[:-1], groups[1:]))


def get_base_model(model_name):
    return re.sub(r"_run\d+$", "", model_name)


def compute_uot_plans(X, t_train, delta=1, use_mini_batch_uot=False, chunk_size=1000, draw=False):

    gamma0_plans = []
    gamma1_plans = []
    sampling_info_plans = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_action = 0

    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):
        x_source, x_target = X[i], X[i + 1]
        n_source, n_target = x_source.shape[0], x_target.shape[0]
        a = np.ones(n_source)
        b = np.ones(n_target)

        norm_2_dist = ot.dist(x_source, x_target, metric="euclidean")
        cos_sq = np.cos(np.minimum(norm_2_dist / (2 * delta), np.pi / 2)) ** 2
        cost_matrix = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))

        if not use_mini_batch_uot:
            a_cuda = torch.from_numpy(a).to(device)
            b_cuda = torch.from_numpy(b).to(device)
            cost_matrix_cuda = torch.from_numpy(cost_matrix).to(device)
            G = ot.unbalanced.mm_unbalanced(a_cuda, b_cuda, cost_matrix_cuda, reg_m=[1.0, 1.0])
            total_action += 2 * (delta**2) * ot.unbalanced.mm_unbalanced2(
                a_cuda, b_cuda, cost_matrix_cuda, reg_m=[1.0, 1.0], returnCost="total"
            )
            G = G.cpu().numpy()
            sampling_info_plans.append(None)
        else:
            group_number = n_source // chunk_size + 1
            G = np.zeros((n_source, n_target))
            source_perm = np.arange(n_source)
            np.random.shuffle(source_perm)
            target_perm = np.arange(n_target)
            np.random.shuffle(target_perm)
            source_indices_groups = np.array_split(source_perm, group_number)
            target_indices_groups = np.array_split(target_perm, group_number)

            gamma0_sub_plans = []
            for src_idx, tgt_idx in zip(source_indices_groups, target_indices_groups):
                sub_cost_matrix = cost_matrix[np.ix_(src_idx, tgt_idx)]
                sub_a = a[src_idx]
                sub_b = b[tgt_idx]
                sub_a_cuda = torch.from_numpy(sub_a).to(device)
                sub_b_cuda = torch.from_numpy(sub_b).to(device)
                sub_cost_matrix = torch.from_numpy(sub_cost_matrix).to(device)
                G_sub = ot.unbalanced.mm_unbalanced(sub_a_cuda, sub_b_cuda, sub_cost_matrix, reg_m=[1.0, 1.0])
                G_sub = G_sub.cpu().numpy()
                G[np.ix_(src_idx, tgt_idx)] = G_sub

                g_sub_sum_1 = G_sub.sum(1)
                gamma0_sub = ((sub_a / (g_sub_sum_1 + 1e-12))[:, None]) * G_sub
                gamma0_sub_plans.append(gamma0_sub.astype(np.float32))

            sampling_info = {
                "sub_plans": gamma0_sub_plans,
                "source_groups": source_indices_groups,
                "target_groups": target_indices_groups,
            }
            sampling_info_plans.append(sampling_info)

        g_sum_1 = G.sum(1)
        g_sum_0 = G.sum(0)
        gamma0_plan = ((a / (g_sum_1 + 1e-12))[:, None]) * G
        gamma1_plan = (b / (g_sum_0 + 1e-12)) * G

        gamma0_plans.append(gamma0_plan)
        gamma1_plans.append(gamma1_plan)

    print(f"total_action: {total_action / X[0].shape[0]}")
    return gamma0_plans, gamma1_plans, sampling_info_plans


def sample_from_ot_plan(ot_plan,  # WFR-OET coupling
                        x0,  # 初始状态 t0 表达
                        x1,  # 目标状态 t1 表达
                        batch_size,
                        sampling_info=None):  # mini-batch OET的分组信息

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    if sampling_info is None:
        pi_cuda = torch.from_numpy(ot_plan.astype(np.float32)).to(device)
        row_sums = pi_cuda.sum(axis=1)
        total_sum = row_sums.sum()
        if total_sum < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        row_probs = row_sums / total_sum
        i_samples = torch.multinomial(row_probs, num_samples=batch_size, replacement=True)
        selected_rows = pi_cuda[i_samples]
        selected_row_sums = row_sums[i_samples]
        conditional_probs = selected_rows / (selected_row_sums.unsqueeze(1) + 1e-12)
        j_samples = torch.multinomial(conditional_probs, num_samples=1).squeeze(1)
        i, j = i_samples.cpu().numpy(), j_samples.cpu().numpy()
    else:
        g_subs = sampling_info["sub_plans"]
        source_indices_groups = sampling_info["source_groups"]
        target_indices_groups = sampling_info["target_groups"]
        block_masses = [g.sum() for g in g_subs]
        total_mass = sum(block_masses)
        if total_mass < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        block_probs = torch.tensor(block_masses, dtype=torch.float32, device=device) / total_mass
        sampled_group_indices = torch.multinomial(block_probs, num_samples=batch_size, replacement=True)
        g_subs_gpu = [torch.from_numpy(g).to(device) for g in g_subs]
        source_indices_gpu = [torch.from_numpy(idx).to(device) for idx in source_indices_groups]
        target_indices_gpu = [torch.from_numpy(idx).to(device) for idx in target_indices_groups]
        unique_groups, counts = torch.unique(sampled_group_indices, return_counts=True)
        final_i_samples = torch.empty(batch_size, dtype=torch.int64, device=device)
        final_j_samples = torch.empty(batch_size, dtype=torch.int64, device=device)
        for group_idx, count in zip(unique_groups, counts):
            g_sub = g_subs_gpu[group_idx]
            sub_row_sums = g_sub.sum(axis=1)
            if sub_row_sums.sum() < 1e-9:
                continue
            sub_row_probs = sub_row_sums / sub_row_sums.sum()
            i_local = torch.multinomial(sub_row_probs, num_samples=count.item(), replacement=True)
            selected_sub_rows = g_sub[i_local]
            selected_sub_row_sums = sub_row_sums[i_local]
            sub_cond_probs = selected_sub_rows / (selected_sub_row_sums.unsqueeze(1) + 1e-12)
            j_local = torch.multinomial(sub_cond_probs, num_samples=1).squeeze(1)
            global_i = source_indices_gpu[group_idx][i_local]
            global_j = target_indices_gpu[group_idx][j_local]
            mask = sampled_group_indices == group_idx
            final_i_samples[mask] = global_i
            final_j_samples[mask] = global_j
        i, j = final_i_samples.cpu().numpy(), final_j_samples.cpu().numpy()

    if i.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return x0[i], x1[j], i, j


def compute_xt_ut_gt(t_relative,
                     delta_t,
                     x0,
                     x1,
                     mass0,
                     mass1,
                     delta):
    index = torch.norm(x1 - x0, dim=1) < torch.pi * delta

    t_relative = t_relative[index]
    x0 = x0[index]
    x1 = x1[index]
    mass0 = mass0[index]
    mass1 = mass1[index]

    diff = x1 - x0
    norm = torch.norm(diff, dim=1, keepdim=True)
    norm_vector = diff / (norm + 1e-9)

    tau = torch.tan(norm / (2 * delta))
    scale = torch.sqrt(mass0 * mass1 / (1 + tau**2))
    omega = 2 * delta * tau * scale
    omega_vector = omega * norm_vector

    A = mass1 + mass0 - 2 * scale
    B = mass0 - scale
    inv_sqrt_am0_m_bsq = 2 * delta / (omega + 1e-9)

    xt_samp = x0 + omega_vector * (
        inv_sqrt_am0_m_bsq
        * (
            torch.arctan((A * t_relative - B) * inv_sqrt_am0_m_bsq)
            - torch.arctan(-B * inv_sqrt_am0_m_bsq)
        )
    )

    masst_samp = A * t_relative**2 - 2 * B * t_relative + mass0
    dmasst_dt = 2 * A * t_relative - 2 * B
    gt_samp = dmasst_dt / masst_samp * (1 / delta_t)
    ut_samp = omega_vector / masst_samp * (1 / delta_t)

    return xt_samp, gt_samp, ut_samp, masst_samp / mass0, index


def get_batch(X,
              t_train,
              batch_size,
              gamma0_plans,  # γ0
              gamma1_plans,  # γ1
              delta,
              ratios,
              sampling_info_plans):

    ts = []
    xts = []
    uts = []
    gts = []
    massts = []

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    for t in range(len(t_train) - 1):

        # 获取当前时间对的耦合
        gamma0_plan = gamma0_plans[t]
        gamma1_plan = gamma1_plans[t]
        # mini-batch 信息
        sampling_info = sampling_info_plans[t]

        # 从 γ₀ 采样条件对 (x₀, x₁)
        x0, x1, idx_0, idx_1 = sample_from_ot_plan(gamma0_plan, X[t], X[t + 1], batch_size, sampling_info)
        
        x0 = torch.from_numpy(x0).float().to(device)
        x1 = torch.from_numpy(x1).float().to(device)

        # 获取端点质量: mass₀=γ₀(x₀,x₁), mass₁=γ₁(x₀,x₁)
        mass0 = torch.from_numpy(gamma0_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        mass1 = torch.from_numpy(gamma1_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)

        # 计算时间间隔和随机相对时间
        delta_t = t_train[t + 1] - t_train[t]
        t_relative = torch.rand(x0.shape[0], 1).type_as(x0)
        t_samp = delta_t * t_relative  # 实际时间偏移: τ = t₀ + s·Δt

        # 计算目标场 (u_t, g_t)
        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(
            t_relative, delta_t, x0, x1, mass0, mass1, delta
        )

        ts.append(t_samp[index] + t_train[t])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)

    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(gts), torch.cat(massts)


def compute_trajectories_action(df, f_net, device, all_times, sigma, use_mass, delta, num_points=None, num_runs=1):
    del sigma
    start_time = float(np.min(all_times))
    end_time = float(np.max(all_times))
    data = torch.tensor(df[df["samples"] == start_time].values, dtype=torch.float32).requires_grad_()
    x0 = data[:, 1:].to(device).requires_grad_()

    results = []
    for run_idx in tqdm(range(num_runs)):
        seed = run_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if num_points is not None:
            sample_indices = random.sample(range(x0.size(0)), num_points)
            x0_subset = x0[sample_indices].to(device)
        else:
            x0_subset = x0.to(device)

        lnw0 = torch.log(torch.ones(x0_subset.shape[0], 1)).to(device)
        initial_state = (x0_subset, lnw0)

        for param in f_net.parameters():
            param.requires_grad = False

        ts = torch.linspace(start_time, end_time, 100, device=device)
        traj, traj_lnw = odeint(ODEFunc2(f_net), initial_state, ts)

        if use_mass:
            weight = torch.exp(traj_lnw)
        else:
            weight = torch.ones_like(traj_lnw)

        dt = ts[1:] - ts[:-1]
        action = 0.0
        for time_idx in range(len(ts) - 1):
            this_t = ts[time_idx]
            this_x = traj[time_idx]
            this_m = weight[time_idx]
            this_vel = f_net.v_net(this_t, this_x)
            loss_vel = 0.5 * torch.sum((this_vel**2).sum(dim=1, keepdim=True) * this_m)

            this_g = f_net.g_net(this_t, this_x)
            loss_g = 0.5 * (delta**2) * torch.sum((this_g**2) * this_m)

            action += (loss_vel + loss_g) * dt[time_idx]

        action_value = action.detach().cpu().item() / x0_subset.shape[0]
        results.append(
            {
                "Model": f"WFR-FM_run{run_idx + 1}",
                "action": action_value,
            }
        )

    return results


###############################################################################################
"""
CLZ:    
此处开始为我们的 Multiscale OT 新增内容
"""
###############################################################################################


def compute_uot_sparse(
    a,
    b,
    M,  # torch.sparse_coo_tensor
    reg_m,
    c=None,
    reg=0.0,
    div="kl",
    G0=None,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
):
    import torch
    from ot.utils import get_parameter_pair

    assert torch.is_tensor(M) and M.is_sparse, "M must be torch sparse COO tensor"

    device = M.device
    dtype = M.dtype

    # ===== coalesce（必须）=====
    M = M.coalesce()
    indices = M.indices()  # [2, nnz]
    row = indices[0]
    col = indices[1]
    M_data = M.values()

    dim_a, dim_b = M.shape

    # ===== a, b =====
    a = torch.as_tensor(a, device=device, dtype=dtype).flatten()
    b = torch.as_tensor(b, device=device, dtype=dtype).flatten()

    if a.numel() == 0:
        a = torch.ones(dim_a, device=device, dtype=dtype) / dim_a
    if b.numel() == 0:
        b = torch.ones(dim_b, device=device, dtype=dtype) / dim_b

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    # ===== 初始化 G =====
    if G0 is None:
        G_data = a[row] * b[col]
    else:
        if G0.is_sparse:
            G_data = G0.coalesce().values()
        else:
            G_data = G0[row, col]

    # ===== Kernel K =====
    div = div.lower()

    if div == "kl":
        sum_r = reg + reg_m1 + reg_m2
        r1 = reg_m1 / sum_r
        r2 = reg_m2 / sum_r
        r = reg / sum_r

        # 数值稳定（防爆）
        scaled_M = torch.clamp(M_data / sum_r, max=50.0)

        K_data = (
            (a[row] ** r1)
            * (b[col] ** r2)
            * torch.exp(-scaled_M)
        )

        if reg > 0:
            if c is None:
                c_data = a[row] * b[col]
            else:
                c = c.coalesce()
                c_data = c.values()
            K_data = K_data * (c_data ** r)

    elif div == "l2":
        K_data = (
            reg_m1 * a[row]
            + reg_m2 * b[col]
            - M_data
        )
        K_data = torch.clamp(K_data, min=0.0)

    else:
        raise ValueError("Unknown div")

    # ===== log =====
    if log:
        log_dict = {"err": []}

    # ===== 主循环 =====
    for i in range(numItermax):
        G_prev = G_data.clone()

        # ===== marginals（关键：scatter）=====
        u = torch.zeros(dim_a, device=device, dtype=dtype)
        v = torch.zeros(dim_b, device=device, dtype=dtype)

        u.index_add_(0, row, G_data)
        v.index_add_(0, col, G_data)

        if div == "kl":
            u_pow = torch.pow(u, r1)
            v_pow = torch.pow(v, r2)

            denom = u_pow[row] * v_pow[col] + 1e-16

            G_power = torch.pow(G_data, r1 + r2)

            G_data = K_data * G_power / denom

        elif div == "l2":
            denom = (
                reg_m1 * u[row]
                + reg_m2 * v[col]
                + reg * G_data
                + 1e-16
            )
            G_data = K_data * G_data / denom

        # ===== 收敛 =====
        err = torch.norm(G_data - G_prev)

        if log:
            log_dict["err"].append(err.item())

        if verbose:
            print(f"{i:5d}|{err:.6e}|")

        if err.item() < stopThr:
            break

    # ===== 输出 sparse G =====
    G = torch.sparse_coo_tensor(
        indices,
        G_data,
        size=(dim_a, dim_b),
        device=device,
        dtype=dtype,
    )

    if log:
        return G.coalesce(), log_dict
    else:
        return G.coalesce()


def get_cluster(df, cluster_col, feature_prefix='x'):
    """
    CLZ:
    此函数用于计算指定尺度 cluster 的质心 (centroids) 和权重 (weights)
    自动排除 samples、scale* 列，仅保留 x1, x2... 特征列

    权重 = 点的数量 / 总点数 (平衡)
        = 点的数量 (非平衡)

    参数 cluster_col: 指定为哪一尺度 (scale0, scale1...)
    """
    # 构建排除列集合（自动排除 samples、scale* 列）
    exclude_cols = {'samples'}
    exclude_cols.update([c for c in df.columns if c.lower().startswith('scale')])

    # 动态识别特征列（按字母排序保证多尺度间维度严格对齐）
    feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix) and c not in exclude_cols])

    if not feature_cols:
        raise ValueError(f"No valid feature columns found with prefix '{feature_prefix}'. Please check column names.")

    # 分组计算
    groups = df.groupby(cluster_col)
    centroids = groups[feature_cols].mean().values  # shape: (n_clusters, n_features)

    counts = groups.size().values
    # weights = counts / counts.sum()
    weights = counts.astype(float)  # 直接用点数作为质量

    ids = groups.groups.keys()  # 保持 ID 顺序

    return centroids, weights, list(ids)


def get_feature(df, feature_prefix='x'):
    """
    CLZ:
    此函数排除 samples、scale* 列，仅保留 x1, x2... 特征列
    """
    # 构建排除列集合（自动排除 samples、scale* 列）
    exclude_cols = {'samples'}
    exclude_cols.update([c for c in df.columns if c.lower().startswith('scale')])

    # 动态识别特征列（按字母排序保证多尺度间维度严格对齐）
    feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix) and c not in exclude_cols])

    if not feature_cols:
        raise ValueError(f"No valid feature columns found with prefix '{feature_prefix}'. Please check column names.")

    # 将特征列数据保存
    feature_data = df[feature_cols]

    return feature_data


def get_data(df, feature_prefix='x'):
    """
    CLZ:
    此函数排除 scale* 列，仅保留 x1, x2... 特征列和 samples 时间点标注列
    与 WFR-FM 使用的数据格式一致
    """
    # 构建排除列集合（自动排除 scale* 列）
    exclude_cols = set([c for c in df.columns if c.lower().startswith('scale')])

    # 动态识别特征列（按字母排序保证多尺度间维度严格对齐）
    feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix) and c not in exclude_cols])

    if not feature_cols:
        raise ValueError(f"No valid feature columns found with prefix '{feature_prefix}'. Please check column names.")

    # 将特征列数据与时间点标注保存
    data_cols = ['samples'] + feature_cols if 'samples' in df.columns else feature_cols
    feature_data = df[data_cols]

    return feature_data


def compute_wfr_oet_cost_matrix(x_src, x_tgt, delta):
    """
    CLZ:
    计算 WFR-OET 代价矩阵: c(x,y) = -log(cos²(min(||x-y||/(2δ), π/2)))
    """
    dist_matrix = ot.dist(x_src, x_tgt, metric='euclidean')
    angle = np.minimum(dist_matrix / (2 * delta), np.pi / 2)
    cos_sq = np.cos(angle) ** 2
    cost = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))

    return cost


def build_macro_to_micro_indices(df, micro_ids_order, id_col='scale1', parent_col='scale0'):
    """
    CLZ:
    构建映射字典: { macro_id: [micro_index_1, micro_index_2, ...] }
    注意: 这里的 value 是 C_micro 数组中的行索引(0, 1, 2...)，不是 micro_id
    """
    # 1. 创建 micro_id 到 数组索引 的映射
    id_to_idx = {mid: i for i, mid in enumerate(micro_ids_order)}

    # 2. 获取每个 micro_id 对应的 macro_id
    # 假设每个 micro_cluster 只属于一个 macro_cluster
    micro_to_macro = df.drop_duplicates(subset=[id_col]).set_index(id_col)[parent_col].to_dict()

    # 3. 分组
    macro_to_indices = {}
    for mid, idx in id_to_idx.items():
        parent = micro_to_macro.get(mid)
        if parent is not None:
            if parent not in macro_to_indices:
                macro_to_indices[parent] = []
            macro_to_indices[parent].append(idx)

    return macro_to_indices


def get_micro_idx_to_point_indices(df, micro_ids_order, cluster_col='scale1'):
    """
    CLZ:
    构建映射: { Level2_Micro_Index : [Level3_Point_Indices] }
    """
    # 1. 获取 df 中每个 micro_id 对应的点索引列表
    # groupby().indices 返回一个字典: {micro_id: array([point_idx_1, point_idx_2, ...])}
    # 这是 Pandas 中获取分组索引最快的方法
    group_indices = df.groupby(cluster_col).indices

    # 2. 将 micro_id 转换为 Level 2 数组中的 index (0, 1, 2...)
    idx_map = {}
    for i, mid in enumerate(micro_ids_order):
        if mid in group_indices:
            idx_map[i] = group_indices[mid]
        else:
            idx_map[i] = []  # 防御性：该 micro 可能没有点

    return idx_map


def get_scale_columns(df):
    """
    CLZ:
    自动获取 data 中的尺度
    """
    return sorted([col for col in df.columns if col.startswith("scale")],
                  key=lambda x: int(x.replace("scale", "")))


"""
def compute_multiscale_wfr_oet_coupling_sparse(
        x_source, x_target,
        df_src, df_tgt,
        delta=1.0,
        reg_m=[1.0, 1.0],
        independent = None
):

    print("Starting hierarchical WFR-OET solving (auto scales)...")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # =========================
    # 1. 自动获取 scale 列
    # =========================
    scale_cols = sorted(
        [c for c in df_src.columns if c.startswith("scale")],
        key=lambda x: int(x.replace("scale", ""))
    )

    n_scales = len(scale_cols)
    print(f"Detected {n_scales} scales:", scale_cols)

    # =========================
    # 2. 存储每层结果
    # =========================
    gammas = []           # 每一层 coupling
    cluster_ids_list = [] # 每层 cluster ids
    centers_list = []
    weights_list = []

    # =========================
    # 3. 逐层计算 OT
    # =========================
    prev_gamma = None

    for level, scale_col in enumerate(scale_cols):
        print(f"--- Running Level {level}: {scale_col} ---")

        C_src, w_src, ids_src = get_cluster(df_src, scale_col)
        C_tgt, w_tgt, ids_tgt = get_cluster(df_tgt, scale_col)

        M = compute_wfr_oet_cost_matrix(C_src, C_tgt, delta)

        # =========================
        # 关键：剪枝（不是第一层）
        # =========================
        if prev_gamma is not None:
            threshold = 1e-8

            # 构建当前层 → 上一层的映射
            parent_col = scale_cols[level - 1]

            src_parent = df_src.groupby(scale_col)[parent_col].first().to_dict()
            tgt_parent = df_tgt.groupby(scale_col)[parent_col].first().to_dict()

            for i, cid_src in enumerate(ids_src):
                p_src = src_parent[cid_src]
                for j, cid_tgt in enumerate(ids_tgt):
                    p_tgt = tgt_parent[cid_tgt]

                    if prev_gamma[p_src, p_tgt] < threshold:
                        M[i, j] = np.inf

        # =========================
        # 求解 OT
        # =========================
        start_time = time.perf_counter()

        gamma = ot.unbalanced.mm_unbalanced(w_src, w_tgt, M, reg_m=reg_m)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Level {level} 运行耗时: {elapsed_time:.6f} 秒")

        print(f"Level {level} done. Nonzero: {(gamma>1e-8).sum()}")

        # 可视化 Level 矩阵 (看看是否有些块被禁用了)
        save_dir = "/home1/clz/北大/MS-UOT-FM/results/multiscale_2d_test/multiscale_plan"
        os.makedirs(save_dir, exist_ok=True)
        scale_name = scale_cols[level]

        plt.figure(figsize=(6, 5))
        plt.imshow(gamma, cmap='hot', interpolation='nearest')
        plt.title(f"Transport Plan - Level {level} ({scale_name})")
        plt.xlabel(f"Target Clusters ({scale_name})")
        plt.ylabel(f"Source Clusters ({scale_name})")
        plt.colorbar()
        save_path = os.path.join(save_dir, f"transport_plan_level{level}_{scale_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        gammas.append(gamma)
        cluster_ids_list.append((ids_src, ids_tgt))
        centers_list.append((C_src, C_tgt))
        weights_list.append((w_src, w_tgt))

        prev_gamma = gamma

    # =========================
    # Final Level → Point（严格等价版）
    # =========================
    print("--- Running Final Point Scale (Sparse COO) ---")

    X_src, X_tgt = x_source, x_target
    n_src, n_tgt = X_src.shape[0], X_tgt.shape[0]

    a = np.ones(n_src)
    b = np.ones(n_tgt)

    # 必须保证顺序完全一致（关键）
    micro_ids_src, micro_ids_tgt = cluster_ids_list[-1]

    map_micro_to_pts_src = get_micro_idx_to_point_indices(df_src, micro_ids_src, cluster_col=scale_cols[-1])
    map_micro_to_pts_tgt = get_micro_idx_to_point_indices(df_tgt, micro_ids_tgt, cluster_col=scale_cols[-1])

    threshold = 1e-8
    active_micro_pairs = np.argwhere(gammas[-1] > threshold)

    rows, cols, data = [], [], []

    for m_src_idx, m_tgt_idx in active_micro_pairs:

        src_pt_indices = map_micro_to_pts_src.get(m_src_idx, [])
        tgt_pt_indices = map_micro_to_pts_tgt.get(m_tgt_idx, [])

        if len(src_pt_indices) == 0 or len(tgt_pt_indices) == 0:
            continue

        block_X_src = X_src[src_pt_indices]
        block_X_tgt = X_tgt[tgt_pt_indices]

        dists = compute_wfr_oet_cost_matrix(block_X_src, block_X_tgt, delta)

        grid_r, grid_c = np.meshgrid(src_pt_indices, tgt_pt_indices, indexing='ij')

        rows.append(grid_r.flatten())
        cols.append(grid_c.flatten())
        data.append(dists.flatten())

    # ===== COO 构建 =====
    if data:
        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_data = np.concatenate(data)

        M_points_sparse = sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(n_src, n_tgt)
        )
    else:
        M_points_sparse = sparse.coo_matrix((n_src, n_tgt))

    print(f"稀疏矩阵构建完成. Shape: {M_points_sparse.shape}")
    print(f"非零元素 (NNZ): {M_points_sparse.nnz}")
    print(f"稀疏度: {M_points_sparse.nnz / (n_src * n_tgt):.4%}")

    print("Solving final point-wise OT with Sparse Matrix...")

    # ===== GPU =====
    w_points_src_cuda = torch.from_numpy(a).to(device)
    w_points_tgt_cuda = torch.from_numpy(b).to(device)

    indices = torch.from_numpy(np.vstack((M_points_sparse.row, M_points_sparse.col))).long().to(device)
    values  = torch.from_numpy(M_points_sparse.data).to(device)
    M_points_sparse_cuda = torch.sparse_coo_tensor(indices, values, M_points_sparse.shape, device=device).coalesce()

    start_time = time.perf_counter()

    gamma_points = compute_uot_sparse(
        w_points_src_cuda, w_points_tgt_cuda,
        M_points_sparse_cuda, reg_m=reg_m
    )

    end_time = time.perf_counter()
    print(f"point ot 代码运行耗时: {end_time - start_time:.6f} 秒")

    return gamma_points
"""


def compute_multiscale_wfr_oet_coupling_sparse(
        x_source, x_target,
        df_src, df_tgt,
        delta=1.0,
        reg_m=[1.0, 1.0],
        independent=True,  # 控制 exact MS-UOT/independent MS-UOT
):
    """
    CLZ:
    多尺度稀疏 WFR-OET 耦合计算主函数
    核心流程: 粗层全连接求解 → 阈值剪枝 → 稀疏代价矩阵组装 → 逐层细化 → 最终点级稀疏求解
    返回: G_points (点级耦合矩阵 [n_src, n_tgt]), sampling_info (供后续采样使用)
    """

    print("Starting hierarchical WFR-OET solving (auto scales)...")

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # =========================
    # 1. 自动获取 scale 列
    # =========================
    scale_cols = sorted(
        [c for c in df_src.columns if c.startswith("scale")],
        key=lambda x: int(x.replace("scale", ""))
    )

    n_scales = len(scale_cols)
    print(f"Detected {n_scales} scales:", scale_cols)

    # =========================
    # 2. 存储每层结果
    # =========================
    gammas = []
    cluster_ids_list = []
    centers_list = []
    weights_list = []

    # =========================
    # 3. 逐层计算 OT
    # =========================
    prev_gamma = None

    for level, scale_col in enumerate(scale_cols):
        print(f"\n--- Running Level {level}: {scale_col} ---")

        C_src, w_src, ids_src = get_cluster(df_src, scale_col)
        C_tgt, w_tgt, ids_tgt = get_cluster(df_tgt, scale_col)

        M = compute_wfr_oet_cost_matrix(C_src, C_tgt, delta)

        if prev_gamma is not None:
            threshold = 1e-8
            parent_col = scale_cols[level - 1]

            src_parent = df_src.groupby(scale_col)[parent_col].first().to_dict()
            tgt_parent = df_tgt.groupby(scale_col)[parent_col].first().to_dict()

            for i, cid_src in enumerate(ids_src):
                p_src = src_parent[cid_src]
                for j, cid_tgt in enumerate(ids_tgt):
                    p_tgt = tgt_parent[cid_tgt]

                    if prev_gamma[p_src, p_tgt] < threshold:
                        M[i, j] = np.inf

        start_time = time.perf_counter()

        gamma = ot.unbalanced.mm_unbalanced(w_src, w_tgt, M, reg_m=reg_m)
        
        end_time = time.perf_counter()
        print(f"Level {level} 运行耗时: {end_time - start_time:.6f} 秒")

        print(f"Level {level} done. Nonzero: {(gamma>1e-8).sum()}")

        # 可视化 Level 矩阵 (看看是否有些块被禁用了)
        save_dir = "/home1/clz/北大/MS-UOT-FM/results/multiscale_2d_test/multiscale_plan"
        os.makedirs(save_dir, exist_ok=True)
        scale_name = scale_cols[level]
        plt.figure(figsize=(6, 5))
        plt.imshow(gamma, cmap='hot', interpolation='nearest')
        plt.title(f"Transport Plan - Level {level} ({scale_name})")
        plt.xlabel(f"Target Clusters ({scale_name})")
        plt.ylabel(f"Source Clusters ({scale_name})")
        plt.colorbar()
        save_path = os.path.join(save_dir, f"transport_plan_level{level}_{scale_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        gammas.append(gamma)
        cluster_ids_list.append((ids_src, ids_tgt))
        prev_gamma = gamma

    # =========================
    # Final Level
    # =========================
    print("\n--- Running Final Point Scale ---")

    X_src, X_tgt = x_source, x_target
    n_src, n_tgt = X_src.shape[0], X_tgt.shape[0]

    micro_ids_src, micro_ids_tgt = cluster_ids_list[-1]

    map_src = get_micro_idx_to_point_indices(df_src, micro_ids_src, cluster_col=scale_cols[-1])
    map_tgt = get_micro_idx_to_point_indices(df_tgt, micro_ids_tgt, cluster_col=scale_cols[-1])

    threshold = 1e-8
    active_pairs = np.argwhere(gammas[-1] > threshold)

    rows, cols, data = [], [], []

    # =========================================================
    # Option 1：Exact Sparse OET
    # =========================================================
    if not independent:

        print("Using Exact Sparse OET at point level")

        for m_src_idx, m_tgt_idx in active_pairs:

            src_pts = map_src.get(m_src_idx, [])
            tgt_pts = map_tgt.get(m_tgt_idx, [])

            if len(src_pts) == 0 or len(tgt_pts) == 0:
                continue

            block_X_src = X_src[src_pts]
            block_X_tgt = X_tgt[tgt_pts]

            dists = compute_wfr_oet_cost_matrix(block_X_src, block_X_tgt, delta)

            grid_r, grid_c = np.meshgrid(src_pts, tgt_pts, indexing='ij')

            rows.append(grid_r.flatten())
            cols.append(grid_c.flatten())
            data.append(dists.flatten())

        if data:
            all_rows = np.concatenate(rows)
            all_cols = np.concatenate(cols)
            all_data = np.concatenate(data)

            M_sparse = sparse.coo_matrix(
                (all_data, (all_rows, all_cols)),
                shape=(n_src, n_tgt)
            )
        else:
            M_sparse = sparse.coo_matrix((n_src, n_tgt))

        w_src = torch.ones(n_src, device=device)
        w_tgt = torch.ones(n_tgt, device=device)

        indices = torch.from_numpy(np.vstack((M_sparse.row, M_sparse.col))).long().to(device)
        values = torch.from_numpy(M_sparse.data).to(device)
        M_sparse_cuda = torch.sparse_coo_tensor(indices, values, M_sparse.shape, device=device).coalesce()

        start_time = time.perf_counter()

        gamma_points = compute_uot_sparse(w_src, w_tgt, M_sparse_cuda, reg_m=reg_m)

        end_time = time.perf_counter()
        print(f"point ot 代码运行耗时: {end_time - start_time:.6f} 秒") 

        return gamma_points

    # =========================================================
    # Option 2：Scalable Heuristic
    # =========================================================
    else:

        print("Using Scalable Heuristic at point level")

        for m_src_idx, m_tgt_idx in active_pairs:

            src_pts = map_src.get(m_src_idx, [])
            tgt_pts = map_tgt.get(m_tgt_idx, [])

            if len(src_pts) == 0 or len(tgt_pts) == 0:
                continue

            n_s = len(src_pts)
            n_t = len(tgt_pts)

            # mass-weighted factorization
            gamma_micro = gammas[-1]
            val = gamma_micro[m_src_idx, m_tgt_idx] / (n_s * n_t)

            grid_r, grid_c = np.meshgrid(src_pts, tgt_pts, indexing='ij')

            rows.append(grid_r.flatten())
            cols.append(grid_c.flatten())
            data.append(np.full(n_s * n_t, val))

        if data:
            all_rows = np.concatenate(rows)
            all_cols = np.concatenate(cols)
            all_data = np.concatenate(data)

            indices = torch.from_numpy(
                np.vstack((all_rows, all_cols))
            ).long().to(device)

            values = torch.from_numpy(all_data).float().to(device)

            gamma_points = torch.sparse_coo_tensor(indices, values, (n_src, n_tgt), device=device).coalesce()
        else:
            gamma_points = torch.sparse_coo_tensor(torch.zeros((2,0), dtype=torch.long, device=device), torch.zeros(0, device=device), (n_src, n_tgt))

        return gamma_points


def compute_multiscale_uot_plans(df, X, t_train, delta=1, use_mini_batch_uot=False, chunk_size=1000):
    """
    CLZ:
    use_mini_batch_uot=False 时，采用 multiscale ot
    use_mini_batch_uot=True 时，仍使用 mini-batch ot
    """
    gamma0_plans = []
    gamma1_plans = []
    sampling_info_plans = []

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    total_action = 0

    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):
        # point 级别
        x_source, x_target = X[i], X[i + 1]
        n_source, n_target = x_source.shape[0], x_target.shape[0]
        a = np.ones(n_source)
        b = np.ones(n_target)

        # 计算 WFR-OET cost
        norm_2_dist = ot.dist(x_source, x_target, metric="euclidean")
        cos_sq = np.cos(np.minimum(norm_2_dist / (2 * delta), np.pi / 2)) ** 2
        cost_matrix = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))

        if not use_mini_batch_uot:
            print("Computing Multiscale UOT plans...")

            """
            CLZ:
            此处开始按照 multiscale ot 逻辑修改
            """
            # 分离当前时间对数据
            df_src = df[df['samples'] == t_train[i]].copy().reset_index(drop=True)
            df_tgt = df[df['samples'] == t_train[i + 1]].copy().reset_index(drop=True)

            G = compute_multiscale_wfr_oet_coupling_sparse(
                x_source, x_target,
                df_src, df_tgt,
                delta=delta,
                reg_m=[1.0, 1.0],
            )

            # G = G.cpu().to_dense().numpy()

            sampling_info_plans.append(None)

        else:
            print("Computing Mini-batch UOT plans...")

            group_number = n_source // chunk_size + 1
            G = np.zeros((n_source, n_target))
            source_perm = np.arange(n_source)
            np.random.shuffle(source_perm)
            target_perm = np.arange(n_target)
            np.random.shuffle(target_perm)
            source_indices_groups = np.array_split(source_perm, group_number)
            target_indices_groups = np.array_split(target_perm, group_number)

            gamma0_sub_plans = []
            for src_idx, tgt_idx in zip(source_indices_groups, target_indices_groups):
                sub_cost_matrix = cost_matrix[np.ix_(src_idx, tgt_idx)]
                sub_a = a[src_idx]
                sub_b = b[tgt_idx]
                sub_a_cuda = torch.from_numpy(sub_a).to(device)
                sub_b_cuda = torch.from_numpy(sub_b).to(device)
                sub_cost_matrix = torch.from_numpy(sub_cost_matrix).to(device)
                G_sub = ot.unbalanced.mm_unbalanced(sub_a_cuda, sub_b_cuda, sub_cost_matrix, reg_m=[1.0, 1.0])
                G_sub = G_sub.cpu().numpy()
                G[np.ix_(src_idx, tgt_idx)] = G_sub

                g_sub_sum_1 = G_sub.sum(1)
                gamma0_sub = ((sub_a / (g_sub_sum_1 + 1e-12))[:, None]) * G_sub
                gamma0_sub_plans.append(gamma0_sub.astype(np.float32))

            sampling_info = {
                "sub_plans": gamma0_sub_plans,
                "source_groups": source_indices_groups,
                "target_groups": target_indices_groups,
            }
            sampling_info_plans.append(sampling_info)

        # 计算 semi-coupling
        # g_sum_1 = G.sum(1)
        # g_sum_0 = G.sum(0)
        # gamma0_plan = ((a / (g_sum_1 + 1e-12))[:, None]) * G
        # gamma1_plan = (b / (g_sum_0 + 1e-12)) * G

        """
        CLZ:    
        OET coupling 算 semi coupling 也要按照稀疏逻辑计算
        """
        indices = G.indices()
        row = indices[0]
        col = indices[1]
        values = G.values()

        device = values.device
        dtype = values.dtype

        n_source, n_target = G.shape

        g_sum_1 = torch.zeros(n_source, device=device, dtype=dtype)
        g_sum_1.index_add_(0, row, values)

        g_sum_0 = torch.zeros(n_target, device=device, dtype=dtype)
        g_sum_0.index_add_(0, col, values)

        # ===== scaling =====
        eps = 1e-12
        a_torch = torch.as_tensor(a, device=device, dtype=dtype)
        b_torch = torch.as_tensor(b, device=device, dtype=dtype)

        scale_row = a_torch[row] / (g_sum_1[row] + eps)
        scale_col = b_torch[col] / (g_sum_0[col] + eps)

        gamma0_values = scale_row * values
        gamma1_values = scale_col * values

        gamma0_plan = torch.sparse_coo_tensor(indices, gamma0_values, G.shape).coalesce()
        gamma1_plan = torch.sparse_coo_tensor(indices, gamma1_values, G.shape).coalesce()

        gamma0_plans.append(gamma0_plan)
        gamma1_plans.append(gamma1_plan)

    print(f"total_action: {total_action / X[0].shape[0]}")

    return gamma0_plans, gamma1_plans, sampling_info_plans


def sample_from_ot_plan_sparse(ot_plan, x0, x1, batch_size, sampling_info=None):
    # ot_plan: sparse_coo_tensor

    # 强制统一：numpy → torch
    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0).float()
    if isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1).float()

    x0 = x0.to(ot_plan._values().device)
    x1 = x1.to(ot_plan._values().device)

    indices = ot_plan._indices()
    values = ot_plan._values()

    eps = 1e-12
    total = values.sum()
    if total < eps:
        empty = torch.empty(0, dtype=torch.long, device=x0.device)
        return x0[:0], x1[:0], empty, empty

    probs = values / (total + eps)

    k_samples = torch.multinomial(probs, batch_size, replacement=True)

    i = indices[0, k_samples]
    j = indices[1, k_samples]

    x0_batch = x0.index_select(0, i)
    x1_batch = x1.index_select(0, j)

    return x0_batch, x1_batch, i, j, k_samples


def get_batch_sparse(X,
              t_train,
              batch_size,
              gamma0_plans,  # γ0
              gamma1_plans,  # γ1
              delta,
              ratios,
              sampling_info_plans):

    ts = []
    xts = []
    uts = []
    gts = []
    massts = []

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    for t in range(len(t_train) - 1):

        # 获取当前时间对的耦合
        gamma0_plan = gamma0_plans[t]
        gamma1_plan = gamma1_plans[t]
        # mini-batch 信息
        sampling_info = sampling_info_plans[t]

        # 从 γ₀ 采样条件对 (x₀, x₁)
        x0, x1, idx_0, idx_1, k_samples = sample_from_ot_plan_sparse(gamma0_plan, X[t], X[t + 1], batch_size, sampling_info)

        # 获取端点质量: mass₀=γ₀(x₀,x₁), mass₁=γ₁(x₀,x₁)
        # mass0 = torch.from_numpy(gamma0_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        # mass1 = torch.from_numpy(gamma1_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        values0 = gamma0_plan._values().float()
        values1 = gamma1_plan._values().float()

        mass0 = values0[k_samples].unsqueeze(-1)
        mass1 = values1[k_samples].unsqueeze(-1)

        # 计算时间间隔和随机相对时间
        delta_t = t_train[t + 1] - t_train[t]
        t_relative = torch.rand(x0.shape[0], 1).type_as(x0)
        t_samp = delta_t * t_relative  # 实际时间偏移: τ = t₀ + s·Δt

        # 计算目标场 (u_t, g_t)
        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(
            t_relative, delta_t, x0, x1, mass0, mass1, delta
        )

        ts.append(t_samp[index] + t_train[t])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)

    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(gts), torch.cat(massts)
