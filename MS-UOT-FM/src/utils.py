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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

"""
def compute_uot_sparse(
        a,
        b,
        M,
        reg_m,
        c=None,
        reg=0,
        div="kl",
        G0=None,
        numItermax=1000,
        stopThr=1e-15,
        verbose=False,
        log=False,
):

    # CLZ:
    # 此函数用于支持 稀疏计算 unbalance ot

    # 1. 检测是否为稀疏矩阵
    is_sparse = sparse.issparse(M)

    # 统一转换为 numpy 数组处理向量
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    dim_a, dim_b = M.shape

    # 处理空分布
    if len(a) == 0:
        a = np.ones(dim_a) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b) / dim_b

    # 获取参数
    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    # ================= 稀疏矩阵专用逻辑 =================
    if is_sparse:
        # 确保 M 是 COO 格式以便于访问 row, col, data
        if not sparse.isspmatrix_coo(M):
            M = M.tocoo()

        # 初始化 G (与 M 具有相同的稀疏结构)
        if G0 is None:
            G_data = a[M.row] * b[M.col]
            G = sparse.coo_matrix((G_data, (M.row, M.col)), shape=M.shape)
        else:
            if sparse.issparse(G0):
                G = G0.tocoo()
            else:
                G_data = np.asarray(G0)[M.row, M.col]
                G = sparse.coo_matrix((G_data, (M.row, M.col)), shape=M.shape)

        # 初始化 Log
        if log:
            log_dict = {"err": [], "G": []}

        div = div.lower()

        # 预处理参考分布 c
        if reg > 0:
            if c is None:
                c_data = a[M.row] * b[M.col]
                c = sparse.coo_matrix((c_data, (M.row, M.col)), shape=M.shape)
            elif sparse.issparse(c):
                c = c.tocoo()
            else:
                c_data = np.asarray(c)[M.row, M.col]
                c = sparse.coo_matrix((c_data, (M.row, M.col)), shape=M.shape)
        else:
            c = None

        # 计算 Kernel K (保持稀疏结构)
        if div == "kl":
            sum_r = reg + reg_m1 + reg_m2
            r1, r2, r = reg_m1 / sum_r, reg_m2 / sum_r, reg / sum_r

            term_a = a[M.row] ** r1
            term_b = b[M.col] ** r2
            term_c = c.data ** r if (c is not None and reg > 0) else 1.0
            term_m = np.exp(-M.data / sum_r)

            K_data = term_a * term_b * term_c * term_m
            K = sparse.coo_matrix((K_data, (M.row, M.col)), shape=M.shape)

        elif div == "l2":
            term_a = reg_m1 * a[M.row]
            term_b = reg_m2 * b[M.col]
            term_c = reg * c.data if (c is not None and reg > 0) else 0.0

            K_data = term_a + term_b + term_c - M.data
            K_data = np.maximum(K_data, 0)
            K = sparse.coo_matrix((K_data, (M.row, M.col)), shape=M.shape)
        else:
            raise ValueError("Unknown div = {}. Must be either 'kl' or 'l2'".format(div))

        # 转换为 CSR 格式以加速矩阵运算
        K = K.tocsr()
        G = G.tocsr()

        # 迭代优化
        for i in range(numItermax):
            Gprev = G

            if div == "kl":
                # 计算边际 (返回稠密向量)
                u = np.array(G.sum(axis=1)).flatten() ** r1
                v = np.array(G.sum(axis=0)).flatten() ** r2

                # G = K * G^(r1+r2) / (u * v)
                G_power = G.power(r1 + r2)
                G_temp = K.multiply(G_power)

                # 行列缩放 (避免构建稠密外积)
                u_safe = np.where(u > 0, 1.0 / u, 0.0)
                v_safe = np.where(v > 0, 1.0 / v, 0.0)

                # G = sparse.diags(u_safe) @ G_temp
                # G = G @ sparse.diags(v_safe)

                # 列缩放 (CSR 有 .indices 属性)
                G_temp.data *= v_safe[G_temp.indices]

                # 行缩放 (需要通过 indptr 构造行索引)
                if G_temp.nnz > 0:  # 避免空矩阵报错
                    row_indices = np.repeat(np.arange(dim_a), np.diff(G_temp.indptr))
                    G_temp.data *= u_safe[row_indices]

                G = G_temp  # G 指向新对象，Gprev 仍指向旧对象

            elif div == "l2":
                u = np.array(G.sum(axis=1)).flatten()
                v = np.array(G.sum(axis=0)).flatten()

                # 计算 Gd 的非零元素值
                Gd_data = reg_m1 * u[G.row] + reg_m2 * v[G.col] + reg * G.data + 1e-16

                # 更新 G.data
                G.data = K.data * G.data / Gd_data
                G = sparse.csr_matrix(G)

            # 收敛性检查
            diff = G - Gprev
            err = np.sqrt(diff.power(2).sum())

            if log:
                log_dict["err"].append(err)
                log_dict["G"].append(G)

            if verbose:
                print("{:5d}|{:8e}|".format(i, err))

            if err < stopThr:
                break

        # 计算最终成本
        if log:
            linear_cost = G.multiply(M).sum()

            m1 = np.array(G.sum(axis=1)).flatten()
            m2 = np.array(G.sum(axis=0)).flatten()

            if div == "kl":
                cost_m1 = np.sum(m1 * np.log(m1 / a + 1e-16) - m1 + a)
                cost_m2 = np.sum(m2 * np.log(m2 / b + 1e-16) - m2 + b)
                cost = linear_cost + reg_m1 * cost_m1 + reg_m2 * cost_m2
                if reg > 0:
                    mask = G.data > 0
                    cost_reg = np.sum(
                        G.data[mask] * np.log(G.data[mask] / (c.data[mask] + 1e-16)) - G.data[mask] + c.data[mask])
                    cost += reg * cost_reg
            else:
                cost = (
                        linear_cost
                        + reg_m1 * 0.5 * np.sum((m1 - a) ** 2)
                        + reg_m2 * 0.5 * np.sum((m2 - b) ** 2)
                )
                if reg > 0:
                    cost += reg * 0.5 * ((G - c).power(2).sum())

            log_dict["cost"] = linear_cost
            log_dict["total_cost"] = cost
            return G, log_dict
        else:
            return G

    else:
        # 如果不是稀疏矩阵，回退到原始实现
        return ot.unbalanced.mm_unbalanced(a, b, M, reg_m, c, reg, div, G0, numItermax, stopThr, verbose, log)
"""

"""
def compute_uot_sparse(
        a,
        b,
        M,
        reg_m,
        c=None,
        reg=0,
        div="kl",
        G0=None,
        numItermax=1000,
        stopThr=1e-15,
        verbose=False,
        log=False,
):

    # 1. 获取统一后端 (自动识别输入所属框架与设备)
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    is_sparse = sparse.issparse(M) or (isinstance(M, torch.Tensor) and M.is_sparse)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if reg > 0:  # regularized case
        c = a[:, None] * b[None, :] if c is None else c
    else:  # unregularized case
        c = 0

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    # ================= 稀疏矩阵专用逻辑 =================
    if is_sparse:

        M = M.coalesce()  # 确保索引规范
        row_idx, col_idx = M.indices()[0], M.indices()[1]

        if G0 is None:
            # 按 M 的非零结构初始化 G，值设为边际外积 a[i]*b[j]
            init_vals = a[row_idx] * b[col_idx]
            G = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), init_vals, M.shape)
        else:
            if G0.is_sparse:
                G = G0.coalesce()
            else:
                # 若传入的 G0 是稠密张量，按 M 的索引提取值转为稀疏
                G = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), G0[row_idx, col_idx], M.shape)

        G = G.coalesce()

        if log:
            log_dict = {"err": [], "G": []}

        div = div.lower()

        if reg > 0:
            if c is None:
                c_data = a[M.row] * b[M.col]
                c = sparse.coo_matrix((c_data, (M.row, M.col)), shape=M.shape)
            elif sparse.issparse(c):
                c = c.tocoo()
            else:
                c_data = nx.asarray(c)[M.row, M.col]
                c = sparse.coo_matrix((c_data, (M.row, M.col)), shape=M.shape)
        else:
            c = None

        # 1. 确保 M 是 coalesced 状态（索引有序且无重复），并提取行列索引与值
        M = M.coalesce()
        row_idx = M.indices()[0]  # shape: (nnz,)
        col_idx = M.indices()[1]  # shape: (nnz,)
        m_vals  = M.values()      # shape: (nnz,)

        # 安全提取 c 的值（兼容稀疏/稠密/None）
        c_val = c.values() if (c is not None and c.is_sparse) else c

        if div == "kl":
            sum_r = reg + reg_m1 + reg_m2
            r1, r2, r = reg_m1 / sum_r, reg_m2 / sum_r, reg / sum_r

            term_a = a[row_idx] ** r1
            term_b = b[col_idx] ** r2
            term_c = c_val ** r if (c is not None and reg > 0) else 1.0
            term_m = torch.exp(-m_vals / sum_r)

            K_data = term_a * term_b * term_c * term_m
            K = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), K_data, M.shape)

            K = K.coalesce() 

        elif div == "l2":
            term_a = reg_m1 * a[row_idx]
            term_b = reg_m2 * b[col_idx]
            term_c = reg * c_val if (c is not None and reg > 0) else 0.0

            K_data = term_a + term_b + term_c - m_vals
            K_data = torch.clamp(K_data, min=0.0)  # 替代 nx.maximum(K_data, 0)

            K = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), K_data, M.shape)

            K = K.coalesce() 

        else:
            raise ValueError("Unknown div = {}. Must be either 'kl' or 'l2'".format(div))
        

        for i in range(numItermax):
            Gprev = G

            # 强制稠密边际向量（解决 layout 冲突）
            device = G.device
            dtype = G.values().dtype
            m1 = torch.zeros(dim_a, dtype=dtype, device=device)
            m2 = torch.zeros(dim_b, dtype=dtype, device=device)

            if div == "kl":
                m1 = m1.index_add_(0, G.indices()[0], G.values())  # 按行索引聚合
                m2 = m2.index_add_(0, G.indices()[1], G.values())  # 按列索引聚合

                u = m1 ** r1
                v = m2 ** r2
                G_power = G.values() ** (r1 + r2)
                G_temp = K.values() * G_power
                u_safe = nx.where(u > 0, 1.0 / u, 0.0)
                v_safe = nx.where(v > 0, 1.0 / v, 0.0)
                
                # COO 结构已固定，直接通过索引对齐广播
                row_indices = G.indices()[0]
                col_indices = G.indices()[1]
                G_temp = G_temp * v_safe[col_indices]
                G_temp = G_temp * u_safe[row_indices]
                G = torch.sparse_coo_tensor(G.indices(), G_temp, G.shape).coalesce()

            elif div == "l2":
                m1 = m1.index_add_(0, G.indices()[0], G.values())  # 按行索引聚合
                m2 = m2.index_add_(0, G.indices()[1], G.values())  # 按列索引聚合

                row_indices = G.indices()[0]
                col_indices = G.indices()[1]
                Gd_data = reg_m1 * m1[row_indices] + reg_m2 * m2[col_indices] + reg * G.values() + 1e-16
                G = torch.sparse_coo_tensor(G.indices(), K.values() * G.values() / Gd_data, G.shape).coalesce()

            diff = G.values() - Gprev.values()
            err = nx.sqrt(nx.sum(diff ** 2))

            if log:
                log_dict["err"].append(err)
                log_dict["G"].append(G)
            if verbose:
                print("{:5d}|{:8e}|".format(i, err))
            if err < stopThr:
                break

        if log:
            linear_cost = nx.sum(G.values() * M.values())
            m1 = m1.index_add_(0, G.indices()[0], G.values())  # 按行索引聚合
            m2 = m2.index_add_(0, G.indices()[1], G.values())  # 按列索引聚合

            if div == "kl":
                cost_m1 = nx.sum(m1 * nx.log(m1 / a + 1e-16) - m1 + a)
                cost_m2 = nx.sum(m2 * nx.log(m2 / b + 1e-16) - m2 + b)
                cost = linear_cost + reg_m1 * cost_m1 + reg_m2 * cost_m2
                if reg > 0:
                    mask = G.values() > 0
                    c_val = c.values() if (c is not None and c.is_sparse) else c
                    cost_reg = nx.sum(
                        G.values()[mask] * nx.log(G.values()[mask] / (c_val[mask] + 1e-16)) - G.values()[mask] + c_val[mask])
                    cost += reg * cost_reg
            else:
                cost = (
                        linear_cost
                        + reg_m1 * 0.5 * nx.sum((m1 - a) ** 2)
                        + reg_m2 * 0.5 * nx.sum((m2 - b) ** 2)
                )
                if reg > 0:
                    c_val = c.values() if (c is not None and c.is_sparse) else c
                    cost += reg * 0.5 * nx.sum((G.values() - c_val) ** 2)

            log_dict["cost"] = linear_cost
            log_dict["total_cost"] = cost
            return G, log_dict
        else:
            return G

    else:
        # 稠密情况回退到原版
        return ot.unbalanced.mm_unbalanced(a, b, M, reg_m, c, reg, div, G0, numItermax, stopThr, verbose, log)
"""


"""
def compute_uot_sparse(
    a,
    b,
    M,
    reg_m,
    c=None,
    reg=0,
    div="kl",
    G0=None,
    numItermax=1000,
    stopThr=1e-15,
    verbose=False,
    log=False,
):
    from ot.backend import get_backend
    from ot.utils import get_parameter_pair
    import numpy as np
    from scipy import sparse

    assert sparse.issparse(M), "This version expects sparse M"

    if not sparse.isspmatrix_coo(M):
        M = M.tocoo()

    row = M.row
    col = M.col
    dim_a, dim_b = M.shape

    # backend
    nx = get_backend(a, b, M.data)

    a = nx.asarray(a)
    b = nx.asarray(b)

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    # ===== 初始化 G =====
    if G0 is None:
        G_data = a[row] * b[col]
    else:
        G_data = nx.asarray(G0.data)

    # ===== Kernel K =====
    if div == "kl":
        sum_r = reg + reg_m1 + reg_m2
        r1, r2, r = reg_m1 / sum_r, reg_m2 / sum_r, reg / sum_r

        M_data = nx.asarray(M.data)

        K_data = (
            (a[row] ** r1)
            * (b[col] ** r2)
            * nx.exp(-M_data / sum_r)
        )

    elif div == "l2":
        M_data = nx.asarray(M.data)

        K_data = (
            reg_m1 * a[row]
            + reg_m2 * b[col]
            - M_data
        )
        K_data = nx.maximum(K_data, 0)

    else:
        raise ValueError("Unknown div")

    # ===== iteration =====
    for i in range(numItermax):
        G_prev = G_data

        # ===== marginals =====
        u = nx.zeros(dim_a, type_as=G_data)
        v = nx.zeros(dim_b, type_as=G_data)

        # scatter add
        nx.scatter_add(u, row, G_data)
        nx.scatter_add(v, col, G_data)

        if div == "kl":
            u = u ** r1
            v = v ** r2

            G_power = G_data ** (r1 + r2)

            denom = (u[row] * v[col]) + 1e-16
            G_data = K_data * G_power / denom

        elif div == "l2":
            denom = (
                reg_m1 * u[row]
                + reg_m2 * v[col]
                + reg * G_data
                + 1e-16
            )
            G_data = K_data * G_data / denom

        # ===== error =====
        err = nx.sqrt(nx.sum((G_data - G_prev) ** 2))

        if verbose:
            print(f"{i:5d}|{err:8e}|")

        if err < stopThr:
            break

    # ===== 返回 sparse =====
    from scipy.sparse import coo_matrix
    G = coo_matrix((nx.to_numpy(G_data), (row, col)), shape=(dim_a, dim_b))

    return G
"""


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


def compute_multiscale_wfr_oet_coupling_sparse(
        x_source, x_target,
        df_src, df_tgt,
        delta=1.0,
        reg_m=[1.0, 1.0],
):
    """
    CLZ:
    多尺度稀疏 WFR-OET 耦合计算主函数
    核心流程: 粗层全连接求解 → 阈值剪枝 → 稀疏代价矩阵组装 → 逐层细化 → 最终点级稀疏求解
    返回: G_points (点级耦合矩阵 [n_src, n_tgt]), sampling_info (供后续采样使用)
    """
    print("Starting hierarchical WFR-OET solving...")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #######################################################################################################
    print("--- Running Level 1: Macro Scale OT ---")

    # 获取 macro 质心和权重
    C_macro_src, w_macro_src, macro_ids_src = get_cluster(df_src, 'scale0')
    C_macro_tgt, w_macro_tgt, macro_ids_tgt = get_cluster(df_tgt, 'scale0')
    M_macro = compute_wfr_oet_cost_matrix(C_macro_src, C_macro_tgt, delta)

    start_time = time.perf_counter()

    # 求解 macro ot
    gamma_macro = ot.unbalanced.mm_unbalanced(w_macro_src, w_macro_tgt, M_macro, reg_m=reg_m)
    # gamma_macro = ot.unbalanced.mm_unbalanced(w_macro_src_cuda, w_macro_tgt_cuda, M_macro_cuda, reg_m=reg_m)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"macro ot 代码运行耗时: {elapsed_time:.6f} 秒")

    print("Macro Transport Plan:\n", np.round(gamma_macro, 3))

    #######################################################################################################
    print("\n--- Running Level 2: Micro Scale OT (Sparse COO) ---")

    # 利用提供的辅助函数获取数据
    C_micro_src, w_micro_src, micro_ids_src = get_cluster(df_src, 'scale1')
    C_micro_tgt, w_micro_tgt, micro_ids_tgt = get_cluster(df_tgt, 'scale1')

    M_micro = compute_wfr_oet_cost_matrix(C_micro_src, C_micro_tgt, delta)

    # 我们需要知道每个 micro_cluster 属于哪个 macro_cluster
    # 创建映射: micro_id -> macro_id
    micro_to_macro_src = df_src.groupby('scale1')['scale0'].first().to_dict()
    micro_to_macro_tgt = df_tgt.groupby('scale1')['scale0'].first().to_dict()

    # 遍历 Micro Cost Matrix 的每一个元素
    # 如果对应的 Macro Parent 之间传输量为 0，则将 Micro Cost 设为无穷大
    threshold = 1e-8  # 判定为 0 的阈值

    for i, m_src in enumerate(micro_ids_src):
        parent_src = micro_to_macro_src[m_src]
        for j, m_tgt in enumerate(micro_ids_tgt):
            parent_tgt = micro_to_macro_tgt[m_tgt]
            # 检查上一层级 (Macro) 是否允许传输
            # 注意：这里假设 macro_ids 也是 0, 1 顺序排列的
            if gamma_macro[parent_src, parent_tgt] < threshold:
                M_micro[i, j] = np.inf

    start_time = time.perf_counter()

    gamma_micro = ot.unbalanced.mm_unbalanced(w_micro_src, w_micro_tgt, M_micro, reg_m=reg_m)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"micro ot 代码运行耗时: {elapsed_time:.6f} 秒")

    # 可视化 Level 2 的矩阵 (看看是否有些块被禁用了)
    plt.figure(figsize=(6, 5))
    plt.imshow(gamma_micro, cmap='hot', interpolation='nearest')
    plt.title("Level 2 (Micro) Transport Plan")
    plt.xlabel("Target Micro Clusters")
    plt.ylabel("Source Micro Clusters")
    plt.colorbar()
    plt.savefig("/home1/clz/北大/MS-UOT-FM/results/micro_transport_plan.png", bbox_inches='tight', dpi=600)

    #######################################################################################################
    print("\n--- Running Level 3: Point Scale OT (Sparse COO) ---")

    X_src, X_tgt = x_source, x_target
    n_src, n_tgt = x_source.shape[0], x_target.shape[0]

    a = np.ones(n_src)
    b = np.ones(n_tgt)

    w_points_src = a
    w_points_tgt = b

    # 利用 Level 2 输出的 micro_ids_src/tgt 列表保持顺序一致
    map_micro_to_pts_src = get_micro_idx_to_point_indices(df_src, micro_ids_src)
    map_micro_to_pts_tgt = get_micro_idx_to_point_indices(df_tgt, micro_ids_tgt)

    # 构建稀疏成本矩阵
    threshold = 1e-8
    # 找出上一层级允许传输的连接 (u, v)
    active_micro_pairs = np.argwhere(gamma_micro > threshold)

    rows = []  # 记录全局 Point Src 索引
    cols = []  # 记录全局 Point Tgt 索引
    data = []  # 记录 欧氏距离

    for m_src_idx, m_tgt_idx in active_micro_pairs:
        # 1. 获取该 Micro Cluster 包含的所有点的索引
        src_pt_indices = map_micro_to_pts_src.get(m_src_idx, [])
        tgt_pt_indices = map_micro_to_pts_tgt.get(m_tgt_idx, [])

        if len(src_pt_indices) == 0 or len(tgt_pt_indices) == 0:
            continue

        # 2. 提取坐标块 (Block Extraction)
        # shape: (N_sub_src, D)
        block_X_src = X_src[src_pt_indices]
        # shape: (N_sub_tgt, D)
        block_X_tgt = X_tgt[tgt_pt_indices]

        # 3. 计算块内距离 (Vectorized)
        # shape: (N_sub_src, N_sub_tgt)
        dists = compute_wfr_oet_cost_matrix(block_X_src, block_X_tgt, delta)

        # 4. 生成全局坐标并存储
        # grid_r: 全局源点索引, grid_c: 全局目标点索引
        grid_r, grid_c = np.meshgrid(src_pt_indices, tgt_pt_indices, indexing='ij')

        rows.append(grid_r.flatten())
        cols.append(grid_c.flatten())
        data.append(dists.flatten())

    # 合并并构建 COO 矩阵
    if data:
        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_data = np.concatenate(data)

        # M_points_sparse 只包含允许传输的路径的 Cost
        # 未存储的位置在 OT 求解时会被视为禁区 (或需要在求解器中处理)
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

    # GPU加速
    w_points_src_cuda = torch.from_numpy(w_points_src).to(device)
    w_points_tgt_cuda = torch.from_numpy(w_points_tgt).to(device)

    indices = torch.from_numpy(np.vstack((M_points_sparse.row, M_points_sparse.col))).long().to(device)
    values  = torch.from_numpy(M_points_sparse.data).to(device)
    M_points_sparse_cuda = torch.sparse_coo_tensor(indices, values, M_points_sparse.shape, device=device).coalesce()

    start_time = time.perf_counter()

    gamma_points = compute_uot_sparse(w_points_src_cuda, w_points_tgt_cuda, M_points_sparse_cuda, reg_m=reg_m)

    end_time = time.perf_counter()
    print(f"point ot 代码运行耗时: {end_time - start_time:.6f} 秒")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    return x0_batch, x1_batch, i, j


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

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for t in range(len(t_train) - 1):

        # 获取当前时间对的耦合
        gamma0_plan = gamma0_plans[t]
        gamma1_plan = gamma1_plans[t]
        # mini-batch 信息
        sampling_info = sampling_info_plans[t]

        # 从 γ₀ 采样条件对 (x₀, x₁)
        x0, x1, idx_0, idx_1 = sample_from_ot_plan_sparse(gamma0_plan, X[t], X[t + 1], batch_size, sampling_info)

        # 获取端点质量: mass₀=γ₀(x₀,x₁), mass₁=γ₁(x₀,x₁)
        # mass0 = torch.from_numpy(gamma0_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        # mass1 = torch.from_numpy(gamma1_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)

        gamma0_plan_dense = gamma0_plan.to_dense()
        gamma1_plan_dense = gamma1_plan.to_dense()

        mass0 = gamma0_plan_dense[idx_0, idx_1].to(device).float().unsqueeze(-1)
        mass1 = gamma1_plan_dense[idx_0, idx_1].to(device).float().unsqueeze(-1)    

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
