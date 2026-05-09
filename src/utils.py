import random
import re

import numpy as np
import ot
import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from scipy import sparse
import time

from models import ODEFunc2


def get_base_model(model_name):
    return re.sub(r"_run\d+$", "", model_name)


def get_data(df, feature_prefix='x'):
    """
    仅保留 samples 时间点标注列和 x1, x2, ... 特征列。
    与 WFR-FM 使用的数据格式一致。

    例如：
        输入列: samples, scale0, scale1, prior0, prior1, x1, ..., x10
        输出列: samples, x1, ..., x10
    """
    pattern = re.compile(rf"^{re.escape(feature_prefix)}(\d+)$")

    # 只提取 x1, x2, ..., x10 这种严格格式的特征列
    feature_cols = [
        c for c in df.columns
        if pattern.match(c)
    ]

    if not feature_cols:
        raise ValueError(
            f"No valid feature columns found with prefix '{feature_prefix}'. "
            f"Expected columns like {feature_prefix}1, {feature_prefix}2, ..."
        )

    # 按数字顺序排序，避免 x10 排在 x2 前面
    feature_cols = sorted(
        feature_cols,
        key=lambda c: int(pattern.match(c).group(1))
    )

    data_cols = ['samples'] + feature_cols if 'samples' in df.columns else feature_cols

    return df[data_cols]


def get_data_weighted(df, feature_prefix='x', weight_col='cell_weight'):
    """
    仅保留 samples 时间点标注列、x1, x2, ... 特征列，以及可选的 cell_weight 列。
    自动排除 scale* / prior* 等非特征标注列。
    """
    # 只匹配 x1, x2, ..., x10，避免误选其他以 x 开头的列
    pattern = re.compile(rf"^{re.escape(feature_prefix)}(\d+)$")

    feature_cols = [
        c for c in df.columns
        if pattern.match(c)
    ]

    if not feature_cols:
        raise ValueError(
            f"No valid feature columns found with prefix '{feature_prefix}'. "
            f"Expected columns like {feature_prefix}1, {feature_prefix}2, ..."
        )

    # 按数字顺序排序，避免 x10 排在 x2 前面
    feature_cols = sorted(
        feature_cols,
        key=lambda c: int(pattern.match(c).group(1))
    )

    data_cols = []

    if 'samples' in df.columns:
        data_cols.append('samples')

    data_cols.extend(feature_cols)

    # weighted evaluation 需要保留 cell_weight
    if weight_col in df.columns:
        data_cols.append(weight_col)

    feature_data = df[data_cols]

    return feature_data


def compute_wfr_oet_cost_matrix(x_src, x_tgt, delta):
    """
    计算 WFR-OET 代价矩阵: c(x,y) = -log(cos²(min(||x-y||/(2δ), π/2)))
    """
    dist_matrix = ot.dist(x_src, x_tgt, metric='euclidean')
    angle = np.minimum(dist_matrix / (2 * delta), np.pi / 2)
    cos_sq = np.cos(angle) ** 2
    cost = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))

    return cost


def compute_wfr_distance_unbalanced(gamma, mu_src, mu_tgt, M_points, delta=1.0):
    """
    计算 WFR 函数值
    """
    eps = 1e-10

    gamma_row_sum = gamma.sum(axis=1)
    gamma_col_sum = gamma.sum(axis=0)

    # 直接使用原始质量计算 KL
    # KL(γ_row || μ_src) = ∫ (γ_row * log(γ_row/μ_src) - γ_row + μ_src)
    # 对于离散分布: Σ [γ_row * log(γ_row/μ_src) - γ_row + μ_src]
    kl_src = np.sum(gamma_row_sum * (np.log(gamma_row_sum + eps) - np.log(mu_src + eps))
                    - gamma_row_sum + mu_src)
    kl_tgt = np.sum(gamma_col_sum * (np.log(gamma_col_sum + eps) - np.log(mu_tgt + eps))
                    - gamma_col_sum + mu_tgt)

    integral_term = np.sum(M_points * gamma)

    wfr_squared = 2 * delta ** 2 * (integral_term + kl_src + kl_tgt)

    return wfr_squared, np.sqrt(wfr_squared)


def get_point_weights(df, weight_col="cell_weight"):
    """
    若 df 中存在 cell_weight，则使用其作为每个细胞质量；
    否则默认每个细胞质量为 1。
    """
    if weight_col in df.columns:
        weights = df[weight_col].values.astype(float)

        if np.isnan(weights).any() or np.isinf(weights).any():
            raise ValueError(f"{weight_col} contains NaN or inf.")

        if np.any(weights <= 0):
            raise ValueError(f"{weight_col} contains non-positive values.")

        return weights

    return np.ones(len(df), dtype=float)


def get_cluster(df, cluster_col, feature_prefix='x'):
    """
    计算指定尺度 cluster 的质心和权重。

    若 df 中存在 cell_weight:
        cluster weight = cluster 内 cell_weight 之和
    否则:
        cluster weight = cluster 内细胞数
    """
    pattern = re.compile(rf"^{re.escape(feature_prefix)}(\d+)$")

    feature_cols = [
        c for c in df.columns
        if pattern.match(c)
    ]

    if not feature_cols:
        raise ValueError(
            f"No valid feature columns found with prefix '{feature_prefix}'. "
            f"Expected columns like {feature_prefix}1, {feature_prefix}2, ..."
        )

    feature_cols = sorted(
        feature_cols,
        key=lambda c: int(pattern.match(c).group(1))
    )

    X = df[feature_cols].values.astype(float)
    point_weights = get_point_weights(df)

    groups = df.groupby(cluster_col, sort=True)
    ids = list(groups.groups.keys())

    centroids = []
    weights = []

    for cid in ids:
        idx = groups.indices[cid]
        w = point_weights[idx]

        # 对 MOCA 来说，同一时间点内 cell_weight 相同，
        # 所以加权均值和普通均值等价；这里写成加权形式更鲁棒。
        centroid = np.average(X[idx], axis=0, weights=w)

        centroids.append(centroid)
        weights.append(w.sum())

    centroids = np.vstack(centroids)
    weights = np.asarray(weights, dtype=float)

    return centroids, weights, ids


def get_micro_idx_to_point_indices(df, micro_ids_order, cluster_col):
    """
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


def build_supervised_prior_mask(
        df_src,
        df_tgt,
        scale_col,
        ids_src,
        ids_tgt,
):
    """
    构造 supervised prior mask B^(l)。

    方法：
        source cluster i 和 target cluster j 只有在 prior label 相同的时候才允许传输。

    即：
        B_ij = 1, if prior(src_cluster_i) == prior(tgt_cluster_j)
        B_ij = 0, otherwise
    """

    prior_col = scale_col.replace("scale", "prior", 1)

    if prior_col not in df_src.columns or prior_col not in df_tgt.columns:
        raise ValueError(
            f"Cannot find prior column '{prior_col}' for {scale_col}."
        )

    # 检查：每个 scale cluster 是否只属于唯一 prior
    src_nunique = df_src.groupby(scale_col)[prior_col].nunique()
    tgt_nunique = df_tgt.groupby(scale_col)[prior_col].nunique()

    bad_src = src_nunique[src_nunique > 1]
    bad_tgt = tgt_nunique[tgt_nunique > 1]

    if len(bad_src) > 0:
        raise ValueError(
            f"In source data, some {scale_col} clusters map to multiple {prior_col}: "
            f"{bad_src.head(10).to_dict()}"
        )

    if len(bad_tgt) > 0:
        raise ValueError(
            f"In target data, some {scale_col} clusters map to multiple {prior_col}: "
            f"{bad_tgt.head(10).to_dict()}"
        )

    # scale cluster label -> prior label
    src_prior_map = df_src.groupby(scale_col)[prior_col].first().to_dict()
    tgt_prior_map = df_tgt.groupby(scale_col)[prior_col].first().to_dict()

    src_prior = np.array(
        [src_prior_map[cid] for cid in ids_src]
    )

    tgt_prior = np.array(
        [tgt_prior_map[cid] for cid in ids_tgt]
    )

    # same-prior allowed
    prior_mask = src_prior[:, None] == tgt_prior[None, :]

    return prior_mask, prior_col


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
    """
    稀疏计算 UOT coupling，支持稀疏 cost 输入
    """
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


def compute_multiscale_wfr_oet_coupling_sparse(
        x_source, x_target,
        df_src, df_tgt,
        delta,
        reg_m,
        independent,  # 控制最后一层 OT coupling 方式
        use_supervised_prior,  # 控制先验
        transition_eps=1e-8,
):
    """
    多尺度稀疏 WFR-OET 耦合计算主函数。

    independent=False:
        使用 Exact Sparse OET at point level
        返回 point-level sparse tensor

    independent=True:
        使用文章中的 Scalable Heuristic:
        Masked Independent Matching, Eq.16
        每个细胞权重为 1
        不显式构造 point-level sparse tensor
        返回 sampling_plan dict

    use_supervised_prior=True:
        使用 supervised OT
    """

    print("Starting hierarchical WFR-OET solving (auto scales)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # 1. 自动获取 scale 列
    # =========================
    scale_cols = sorted(
        [c for c in df_src.columns if c.startswith("scale")],
        key=lambda x: int(x.replace("scale", ""))
    )

    df_src = df_src.copy()
    df_tgt = df_tgt.copy()

    # 避免 36.0 / 36 这种 label 映射问题
    for c in scale_cols:
        df_src[c] = df_src[c].astype(int)
        df_tgt[c] = df_tgt[c].astype(int)

    n_scales = len(scale_cols)
    print(f"Detected {n_scales} scales:", scale_cols)

    # =========================
    # 2. 存储每层结果
    # =========================
    gammas = []
    cluster_ids_list = []
    weights_list = []
    masks_list = []

    # =========================
    # 3. 逐层计算 cluster-level OET
    # =========================
    prev_gamma = None

    for level, scale_col in enumerate(scale_cols):
        print(f"\n--- Running Level {level}: {scale_col} ---")

        C_src, w_src, ids_src = get_cluster(df_src, scale_col)
        C_tgt, w_tgt, ids_tgt = get_cluster(df_tgt, scale_col)

        M = compute_wfr_oet_cost_matrix(C_src, C_tgt, delta)

        # 当前层允许转移 mask
        allowed_mask = np.ones(M.shape, dtype=bool)

        # =====================================================
        # Coarse-to-fine hierarchical mask
        # =====================================================
        if prev_gamma is not None:
            parent_col = scale_cols[level - 1]

            prev_ids_src, prev_ids_tgt = cluster_ids_list[-1]
            prev_w_src, prev_w_tgt = weights_list[-1]

            src_parent_label_to_idx = {
                cid: idx for idx, cid in enumerate(prev_ids_src)
            }
            tgt_parent_label_to_idx = {
                cid: idx for idx, cid in enumerate(prev_ids_tgt)
            }

            src_parent = df_src.groupby(scale_col)[parent_col].first().to_dict()
            tgt_parent = df_tgt.groupby(scale_col)[parent_col].first().to_dict()

            for i, cid_src in enumerate(ids_src):
                p_src_label = src_parent[cid_src]

                if p_src_label not in src_parent_label_to_idx:
                    allowed_mask[i, :] = False
                    continue

                p_src_idx = src_parent_label_to_idx[p_src_label]
                denom = max(float(prev_w_src[p_src_idx]), 1e-12)

                for j, cid_tgt in enumerate(ids_tgt):
                    p_tgt_label = tgt_parent[cid_tgt]

                    if p_tgt_label not in tgt_parent_label_to_idx:
                        allowed_mask[i, j] = False
                        continue

                    p_tgt_idx = tgt_parent_label_to_idx[p_tgt_label]

                    # 文章 Eq.9:
                    # P(parent_src -> parent_tgt)
                    # = gamma(parent_src, parent_tgt) / w_source(parent_src)
                    transition_prob = float(prev_gamma[p_src_idx, p_tgt_idx]) / denom

                    if transition_prob < transition_eps:
                        allowed_mask[i, j] = False

        # =====================================================
        # Supervised mask
        # =====================================================
        if use_supervised_prior:
            prior_mask, prior_col = build_supervised_prior_mask(
                df_src=df_src,
                df_tgt=df_tgt,
                scale_col=scale_col,
                ids_src=ids_src,
                ids_tgt=ids_tgt,
            )

            # =====================================================
            # Final mask: M^(l) = B^(l) AND H^(l)
            # =====================================================

            hierarchical_allowed_pairs = int(allowed_mask.sum()) # H^(l)
            prior_allowed_pairs = int(prior_mask.sum())          # B^(l)
            allowed_mask = allowed_mask & prior_mask
            final_allowed_pairs = int(allowed_mask.sum())

            print(
                f"Level {level} prior B^(l) allowed pairs: "
                f"{prior_allowed_pairs} / {prior_mask.size}"
            )
            print(
                f"Level {level} hierarchical H^(l) allowed pairs: "
                f"{hierarchical_allowed_pairs} / {allowed_mask.size}"
            )
            print(
                f"Level {level} final M^(l)=B^(l)&H^(l) allowed pairs: "
                f"{final_allowed_pairs} / {allowed_mask.size}"
            )

        # 禁止转移设为 inf
        M[~allowed_mask] = np.inf

        start_time = time.perf_counter()

        gamma = ot.unbalanced.mm_unbalanced(
            w_src,
            w_tgt,
            M,
            reg_m=reg_m
        )

        end_time = time.perf_counter()

        print(f"Level {level} 运行耗时: {end_time - start_time:.6f} 秒")
        print(f"Level {level} done. Nonzero: {(gamma > 1e-8).sum()}")
        print(f"Level {level} allowed mask pairs: {allowed_mask.sum()}")

        gammas.append(gamma)
        cluster_ids_list.append((ids_src, ids_tgt))
        weights_list.append((w_src, w_tgt))
        masks_list.append(allowed_mask)

        prev_gamma = gamma

    # =========================
    # Final Point Scale
    # =========================
    print("\n--- Running Final Point Scale ---")

    X_src, X_tgt = x_source, x_target
    n_src, n_tgt = X_src.shape[0], X_tgt.shape[0]

    micro_ids_src, micro_ids_tgt = cluster_ids_list[-1]

    map_src = get_micro_idx_to_point_indices(
        df_src,
        micro_ids_src,
        cluster_col=scale_cols[-1]
    )

    map_tgt = get_micro_idx_to_point_indices(
        df_tgt,
        micro_ids_tgt,
        cluster_col=scale_cols[-1]
    )

    # =========================================================
    # Option 1：Exact Sparse OET
    # =========================================================
    if not independent:

        print("Using Exact Sparse OET at point level")

        gamma_micro = np.asarray(gammas[-1], dtype=np.float64)
        final_mask = gamma_micro > transition_eps
        active_pairs = np.argwhere(final_mask)

        rows, cols, data = [], [], []

        for m_src_idx, m_tgt_idx in active_pairs:

            src_pts = map_src.get(m_src_idx, [])
            tgt_pts = map_tgt.get(m_tgt_idx, [])

            if len(src_pts) == 0 or len(tgt_pts) == 0:
                continue

            block_X_src = X_src[src_pts]
            block_X_tgt = X_tgt[tgt_pts]

            dists = compute_wfr_oet_cost_matrix(
                block_X_src,
                block_X_tgt,
                delta
            )

            grid_r, grid_c = np.meshgrid(
                src_pts,
                tgt_pts,
                indexing="ij"
            )

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

        # w_src = torch.ones(n_src, device=device)
        # w_tgt = torch.ones(n_tgt, device=device)
        w_src = torch.as_tensor(
            get_point_weights(df_src),
            device=device,
            dtype=torch.float32
        )

        w_tgt = torch.as_tensor(
            get_point_weights(df_tgt),
            device=device,
            dtype=torch.float32
        )

        indices = torch.from_numpy(
            np.vstack((M_sparse.row, M_sparse.col))
        ).long().to(device)

        values = torch.from_numpy(M_sparse.data).to(device)

        M_sparse_cuda = torch.sparse_coo_tensor(
            indices,
            values,
            M_sparse.shape,
            device=device
        ).coalesce()

        start_time = time.perf_counter()

        gamma_points = compute_uot_sparse(
            w_src,
            w_tgt,
            M_sparse_cuda,
            reg_m=reg_m
        )

        end_time = time.perf_counter()
        print(f"point ot 代码运行耗时: {end_time - start_time:.6f} 秒")

        return gamma_points

    # =========================================================
    # Option 2：Final-Step Within-Block Independent Coupling Lifting
    # Eq.16 + Appendix B
    # =========================================================
    else:

        print("Using Final-Step Within-Block Independent Coupling Lifting")

        # -----------------------------------------------------
        # 文章逻辑：
        #   1. 不在 point level 再求 OET；
        #   2. 使用上一层，即最后一个 annotation scale 的 OET coupling gamma_parent；
        #   3. 先构造 parent-level semi-couplings gamma0_parent/gamma1_parent；
        #   4. 训练采样时隐式 lift 到 point level。
        #
        # 注意：这里沿用你前面要求，point support 直接由 gamma_parent > eps 决定，
        # 不再额外使用 Eq.9 的 gamma / source weight 判断。
        # -----------------------------------------------------
        gamma_parent = np.asarray(gammas[-1], dtype=np.float64)
        w_parent_src, w_parent_tgt = weights_list[-1]
        w_parent_src = np.asarray(w_parent_src, dtype=np.float64)
        w_parent_tgt = np.asarray(w_parent_tgt, dtype=np.float64)

        final_mask = gamma_parent > transition_eps
        n_micro_src, n_micro_tgt = gamma_parent.shape

        # local micro index -> point indices
        src_point_indices = {}
        tgt_point_indices = {}

        micro_src_sizes = np.zeros(n_micro_src, dtype=np.int64)
        micro_tgt_sizes = np.zeros(n_micro_tgt, dtype=np.int64)

        for a in range(n_micro_src):
            pts = np.asarray(map_src.get(a, []), dtype=np.int64)
            src_point_indices[a] = pts
            micro_src_sizes[a] = len(pts)

        for b in range(n_micro_tgt):
            pts = np.asarray(map_tgt.get(b, []), dtype=np.int64)
            tgt_point_indices[b] = pts
            micro_tgt_sizes[b] = len(pts)

        # parent-level row/column sums: s_I, t_J in Appendix B Eq.18
        parent_row_sum = gamma_parent.sum(axis=1)
        parent_col_sum = gamma_parent.sum(axis=0)

        active_pairs = np.argwhere(final_mask)

        block_src = []
        block_tgt = []

        # parent-level semi-coupling mass for each active block
        block_gamma0_mass = []
        block_gamma1_mass = []

        # implicit point-level semi-coupling value inside each block
        # unit-cell weights: alpha=1/n_s, beta=1/n_t
        block_gamma0_value = []
        block_gamma1_value = []

        # rho_IJ = gamma1_parent[I,J] / gamma0_parent[I,J]
        block_m1_ratio = []

        # diagnostics / bookkeeping
        block_parent_gamma = []
        block_parent_row_sum = []
        block_parent_col_sum = []
        skipped_blocks = 0

        eps = 1e-12

        for a, b in active_pairs:
            a = int(a)
            b = int(b)

            n_s = int(micro_src_sizes[a])
            n_t = int(micro_tgt_sizes[b])

            gamma_ab = float(gamma_parent[a, b])
            row_sum_a = float(parent_row_sum[a])
            col_sum_b = float(parent_col_sum[b])

            if (
                n_s <= 0
                or n_t <= 0
                or gamma_ab <= transition_eps
                or row_sum_a <= eps
                or col_sum_b <= eps
            ):
                skipped_blocks += 1
                continue

            # Appendix B Eq.18:
            # gamma0_parent[a,b] = gamma[a,b] / row_sum[a] * w_src[a]
            # gamma1_parent[a,b] = gamma[a,b] / col_sum[b] * w_tgt[b]
            gamma0_parent_ab = gamma_ab / row_sum_a * float(w_parent_src[a])
            gamma1_parent_ab = gamma_ab / col_sum_b * float(w_parent_tgt[b])

            if gamma0_parent_ab <= eps or gamma1_parent_ab <= eps:
                skipped_blocks += 1
                continue

            # Appendix B Eq.22/23 with unit finest-level cell weights:
            # gamma0_ij = gamma0_parent_ab * (1/n_s) * (1/n_t)
            # gamma1_ij = gamma1_parent_ab * (1/n_s) * (1/n_t)
            gamma0_value = gamma0_parent_ab / float(n_s * n_t)
            gamma1_value = gamma1_parent_ab / float(n_s * n_t)
            m1_ratio = gamma1_parent_ab / gamma0_parent_ab

            block_src.append(a)
            block_tgt.append(b)

            block_gamma0_mass.append(gamma0_parent_ab)
            block_gamma1_mass.append(gamma1_parent_ab)

            block_gamma0_value.append(gamma0_value)
            block_gamma1_value.append(gamma1_value)
            block_m1_ratio.append(m1_ratio)

            block_parent_gamma.append(gamma_ab)
            block_parent_row_sum.append(row_sum_a)
            block_parent_col_sum.append(col_sum_b)

        block_src = np.asarray(block_src, dtype=np.int64)
        block_tgt = np.asarray(block_tgt, dtype=np.int64)

        block_gamma0_mass = np.asarray(block_gamma0_mass, dtype=np.float64)
        block_gamma1_mass = np.asarray(block_gamma1_mass, dtype=np.float64)

        block_gamma0_value = np.asarray(block_gamma0_value, dtype=np.float64)
        block_gamma1_value = np.asarray(block_gamma1_value, dtype=np.float64)
        block_m1_ratio = np.asarray(block_m1_ratio, dtype=np.float64)

        block_parent_gamma = np.asarray(block_parent_gamma, dtype=np.float64)
        block_parent_row_sum = np.asarray(block_parent_row_sum, dtype=np.float64)
        block_parent_col_sum = np.asarray(block_parent_col_sum, dtype=np.float64)

        total_gamma0_mass = float(block_gamma0_mass.sum())

        if total_gamma0_mass <= 0:
            raise ValueError(
                "Within-block independent lifting failed: total gamma0 mass is zero. "
                "Check gamma_parent, transition_eps, and cluster point assignments."
            )

        # Appendix B Eq.33: first sample parent block (I,J) ~ gamma0_parent
        block_prob = block_gamma0_mass / total_gamma0_mass
        block_cdf = np.cumsum(block_prob)
        block_cdf[-1] = 1.0

        print("Within-block independent lifting summary:")
        print("  active parent blocks:", len(block_src))
        print("  skipped blocks:", skipped_blocks)
        print("  n_src:", n_src)
        print("  n_tgt:", n_tgt)

        sampling_plan = {
            "type": "implicit_within_block_lift",

            "n_src": n_src,
            "n_tgt": n_tgt,

            "scale_col": scale_cols[-1],

            # local parent index -> original cluster label
            "micro_ids_src": np.asarray(micro_ids_src),
            "micro_ids_tgt": np.asarray(micro_ids_tgt),

            # local parent index -> finest-level point indices
            "src_point_indices": src_point_indices,
            "tgt_point_indices": tgt_point_indices,

            # support induced by the solved parent coupling
            "final_mask": final_mask,

            # parent / child empirical weights
            "micro_src_sizes": micro_src_sizes,
            "micro_tgt_sizes": micro_tgt_sizes,
            "w_parent_src": w_parent_src,
            "w_parent_tgt": w_parent_tgt,

            # parent OET coupling and parent semi-coupling summaries
            "gamma_parent": gamma_parent,
            "parent_row_sum": parent_row_sum,
            "parent_col_sum": parent_col_sum,

            # block sampling representation: block_prob ∝ gamma0_parent[I,J]
            "block_src": block_src,
            "block_tgt": block_tgt,
            "block_prob": block_prob,
            "block_cdf": block_cdf,

            # semi-coupling values / masses after implicit lifting
            "block_gamma0_mass": block_gamma0_mass,
            "block_gamma1_mass": block_gamma1_mass,
            "block_gamma0_value": block_gamma0_value,
            "block_gamma1_value": block_gamma1_value,
            "block_m1_ratio": block_m1_ratio,

            # diagnostics
            "block_parent_gamma": block_parent_gamma,
            "block_parent_row_sum": block_parent_row_sum,
            "block_parent_col_sum": block_parent_col_sum,
        }

        return sampling_plan


def compute_multiscale_uot_plans(
        df,
        X,
        t_train,
        delta,
        use_mini_batch_uot=False,
        independent=True,
        use_supervised_prior=False,
):
    """
    use_mini_batch_uot=False 时，采用 multiscale ot。

    independent=True:
        使用文章 Final-Step Within-Block Independent Coupling Lifting。
        返回 implicit sampling_plan dict。
        gamma0_plans / gamma1_plans 对应位置为 None。

    independent=False:
        使用 Exact Sparse OET。
        返回 sparse tensor G，并构造 gamma0_plan / gamma1_plan。

    use_supervised_prior=True:
        使用 supervised OT
    """

    gamma0_plans = []
    gamma1_plans = []
    sampling_info_plans = []

    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):

        x_source, x_target = X[i], X[i + 1]
        n_source, n_target = x_source.shape[0], x_target.shape[0]

        df_src = (
            df[df["samples"] == t_train[i]]
            .copy()
            .reset_index(drop=True)
        )
        df_tgt = (
            df[df["samples"] == t_train[i + 1]]
            .copy()
            .reset_index(drop=True)
        )

        # 每个细胞权重为 1
        # a = np.ones(n_source)
        # b = np.ones(n_target)
        a = get_point_weights(df_src)
        b = get_point_weights(df_tgt)

        if not use_mini_batch_uot:
            print("Computing Multiscale UOT plans...")

            G = compute_multiscale_wfr_oet_coupling_sparse(
                x_source,
                x_target,
                df_src,
                df_tgt,
                delta=delta,
                reg_m=[1.0, 1.0],
                independent=independent,
                use_supervised_prior=use_supervised_prior,
            )

            # =====================================================
            # Final-Step Within-Block Independent Coupling Lifting:
            # G 是 implicit sampling_plan，不是 sparse tensor
            # =====================================================
            if isinstance(G, dict):
                gamma0_plans.append(None)
                gamma1_plans.append(None)
                sampling_info_plans.append(G)
                continue

            # sampling_info_plans.append(None)

        else:
            print("[Warning] set use_mini_batch=False to compute MS-UOT plans")
            raise NotImplementedError(
                "Mini-batch UOT branch is currently disabled."
            )

        # =========================================================
        # Exact Sparse OET 分支:
        # G 是 sparse tensor，构造 semi-coupling
        # =========================================================

        if not G.is_coalesced():
            G = G.coalesce()

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

        eps = 1e-12

        a_torch = torch.as_tensor(a, device=device, dtype=dtype)
        b_torch = torch.as_tensor(b, device=device, dtype=dtype)

        scale_row = a_torch[row] / (g_sum_1[row] + eps)
        scale_col = b_torch[col] / (g_sum_0[col] + eps)

        gamma0_values = scale_row * values
        gamma1_values = scale_col * values

        gamma0_plan = torch.sparse_coo_tensor(
            indices,
            gamma0_values,
            G.shape,
            device=device
        ).coalesce()

        gamma1_plan = torch.sparse_coo_tensor(
            indices,
            gamma1_values,
            G.shape,
            device=device
        ).coalesce()

        gamma0_plan = gamma0_plan.cpu()
        gamma1_plan = gamma1_plan.cpu()

        # =====================================================
        # 预计算 gamma0 的 CDF，用于快速采样
        # =====================================================
        gamma0_values_for_sampling = gamma0_plan.values()
        gamma0_cdf = torch.cumsum(gamma0_values_for_sampling, dim=0)

        sampling_info = {
            "type": "exact_sparse_cdf",
            "gamma0_cdf": gamma0_cdf,
            "gamma0_total": gamma0_cdf[-1],
            "nnz": gamma0_values_for_sampling.numel(),
        }

        gamma0_plans.append(gamma0_plan)
        gamma1_plans.append(gamma1_plan)
        sampling_info_plans.append(sampling_info)

    return gamma0_plans, gamma1_plans, sampling_info_plans


def sample_from_ot_plan_sparse(ot_plan, x0, x1, batch_size, sampling_info=None):
    """
    从 sparse OT plan 中采样。

    优化点：
    1. 不使用 torch.multinomial，避免 2^24 类别数限制。
    2. 优先使用预计算好的 CDF，避免每个 batch 重复 cumsum。
    """

    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0).float()
    if isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1).float()

    if not ot_plan.is_coalesced():
        ot_plan = ot_plan.coalesce()

    indices = ot_plan.indices()
    values = ot_plan.values()

    device = values.device

    x0 = x0.to(device)
    x1 = x1.to(device)

    eps = 1e-12

    if values.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return x0[:0], x1[:0], empty, empty, empty

    # =====================================================
    # 使用预计算 CDF
    # =====================================================
    if (
        sampling_info is not None
        and isinstance(sampling_info, dict)
        and sampling_info.get("type") == "exact_sparse_cdf"
        and "gamma0_cdf" in sampling_info
    ):
        cdf = sampling_info["gamma0_cdf"]
        total = sampling_info["gamma0_total"]

        # 确保 cdf 和当前 plan 在同一设备
        if cdf.device != device:
            cdf = cdf.to(device)
            total = total.to(device)

            # 更新缓存，避免下次重复搬运
            sampling_info["gamma0_cdf"] = cdf
            sampling_info["gamma0_total"] = total

    else:
        # fallback：没有缓存时才现场计算
        total = values.sum()

        if total < eps:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return x0[:0], x1[:0], empty, empty, empty

        cdf = torch.cumsum(values, dim=0)
        total = cdf[-1]

    if total < eps:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return x0[:0], x1[:0], empty, empty, empty

    # =====================================================
    # CDF + searchsorted 采样
    # =====================================================
    rand = torch.rand(
        batch_size,
        device=device,
        dtype=cdf.dtype
    ) * total

    k_samples = torch.searchsorted(
        cdf,
        rand,
        right=False
    )

    k_samples = torch.clamp(
        k_samples,
        max=values.numel() - 1
    )

    i = indices[0, k_samples]
    j = indices[1, k_samples]

    x0_batch = x0.index_select(0, i)
    x1_batch = x1.index_select(0, j)

    return x0_batch, x1_batch, i, j, k_samples


def sample_from_masked_independent_plan(
        sampling_plan,
        x0,
        x1,
        batch_size,
        device=None,
):
    """
    从论文 Eq.16 / Appendix B 的隐式 finest-level semi-coupling 中采样。

    文章对应逻辑：
        1. parent block: (I, J) ~ gamma0_parent；
        2. source child: i ~ alpha(.|I)，cell-level unit weight 时即在 C0(I) 内均匀采样；
        3. target child: j ~ beta(.|J)，cell-level unit weight 时即在 C1(J) 内均匀采样；
        4. m0 = 1, m1 = rho_IJ = gamma1_parent[I,J] / gamma0_parent[I,J]。

    返回：
        x0_batch, x1_batch, idx_0, idx_1,
        gamma0_value, gamma1_value, m1_ratio
    """

    plan_type = sampling_plan.get("type")
    if plan_type not in {"implicit_within_block_lift", "masked_independent_unit_cell"}:
        raise ValueError(f"Unsupported sampling plan type: {plan_type}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(x0, np.ndarray):
        x0_tensor = torch.from_numpy(x0).float()
    else:
        x0_tensor = x0.float()

    if isinstance(x1, np.ndarray):
        x1_tensor = torch.from_numpy(x1).float()
    else:
        x1_tensor = x1.float()

    x0_tensor = x0_tensor.to(device)
    x1_tensor = x1_tensor.to(device)

    block_prob = np.asarray(sampling_plan["block_prob"], dtype=np.float64)
    num_blocks = len(block_prob)

    if num_blocks == 0:
        raise ValueError("No valid blocks in sampling_plan.")

    prob_sum = float(block_prob.sum())
    if prob_sum <= 0:
        raise ValueError("Invalid sampling_plan: block_prob has zero total mass.")
    if not np.isclose(prob_sum, 1.0):
        block_prob = block_prob / prob_sum

    # 优先使用预计算 CDF，避免 np.random.choice 对超多 block 的额外开销
    block_cdf = sampling_plan.get("block_cdf", None)
    if block_cdf is None:
        block_cdf = np.cumsum(block_prob)
        block_cdf[-1] = 1.0
    else:
        block_cdf = np.asarray(block_cdf, dtype=np.float64)

    rand = np.random.random(size=batch_size)
    sampled_blocks = np.searchsorted(block_cdf, rand, side="left")
    sampled_blocks = np.clip(sampled_blocks, 0, num_blocks - 1)

    idx_0_np = np.empty(batch_size, dtype=np.int64)
    idx_1_np = np.empty(batch_size, dtype=np.int64)

    gamma0_value_np = np.empty(batch_size, dtype=np.float32)
    gamma1_value_np = np.empty(batch_size, dtype=np.float32)
    m1_ratio_np = np.empty(batch_size, dtype=np.float32)

    block_src = sampling_plan["block_src"]
    block_tgt = sampling_plan["block_tgt"]

    src_point_indices = sampling_plan["src_point_indices"]
    tgt_point_indices = sampling_plan["tgt_point_indices"]

    block_gamma0_value = sampling_plan["block_gamma0_value"]
    block_gamma1_value = sampling_plan["block_gamma1_value"]
    block_m1_ratio = sampling_plan["block_m1_ratio"]

    for block_id in np.unique(sampled_blocks):
        pos = np.where(sampled_blocks == block_id)[0]
        k = len(pos)

        a = int(block_src[block_id])
        b = int(block_tgt[block_id])

        src_pts = np.asarray(src_point_indices[a], dtype=np.int64)
        tgt_pts = np.asarray(tgt_point_indices[b], dtype=np.int64)

        if len(src_pts) == 0 or len(tgt_pts) == 0:
            raise ValueError(
                f"Empty block encountered during sampling: "
                f"block_id={block_id}, src_parent={a}, tgt_parent={b}"
            )

        # Eq.33 中 i ~ alpha(.|I), j ~ beta(.|J)。
        # 当前代码 finest-level 元素是单细胞，权重全为 1，因此 alpha/beta 为均匀分布。
        idx_0_np[pos] = np.random.choice(src_pts, size=k, replace=True)
        idx_1_np[pos] = np.random.choice(tgt_pts, size=k, replace=True)

        gamma0_value_np[pos] = block_gamma0_value[block_id]
        gamma1_value_np[pos] = block_gamma1_value[block_id]
        m1_ratio_np[pos] = block_m1_ratio[block_id]

    idx_0 = torch.from_numpy(idx_0_np).long().to(device)
    idx_1 = torch.from_numpy(idx_1_np).long().to(device)

    gamma0_value = torch.from_numpy(gamma0_value_np).float().to(device)
    gamma1_value = torch.from_numpy(gamma1_value_np).float().to(device)
    m1_ratio = torch.from_numpy(m1_ratio_np).float().to(device)

    x0_batch = x0_tensor.index_select(0, idx_0)
    x1_batch = x1_tensor.index_select(0, idx_1)

    return x0_batch, x1_batch, idx_0, idx_1, gamma0_value, gamma1_value, m1_ratio


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


def get_batch_sparse(
        X,
        t_train,
        batch_size,
        gamma0_plans,
        gamma1_plans,
        delta,
        ratios,
        sampling_info_plans
):
    ts = []
    xts = []
    uts = []
    gts = []
    massts = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for t in range(len(t_train) - 1):

        gamma0_plan = gamma0_plans[t]
        gamma1_plan = gamma1_plans[t]
        sampling_info = sampling_info_plans[t]

        # =====================================================
        # Case 1:
        # Final-Step Within-Block Independent Coupling Lifting
        # gamma0_plan 和 gamma1_plan 都是 None
        # sampling_info 是 implicit sampling_plan dict
        # =====================================================
        if gamma0_plan is None and isinstance(sampling_info, dict):

            x0, x1, idx_0, idx_1, gamma0_val, gamma1_val, m1_ratio = (
                sample_from_masked_independent_plan(
                    sampling_info,
                    X[t],
                    X[t + 1],
                    batch_size,
                    device=device,
                )
            )

            # 文章 Algorithm 1:
            # m0 = 1
            # m1 = gamma1 / gamma0
            mass0 = torch.ones_like(m1_ratio).unsqueeze(-1)
            mass1 = m1_ratio.unsqueeze(-1)

        # =====================================================
        # Case 2:
        # Exact Sparse OET
        # gamma0_plan / gamma1_plan 是 sparse tensor
        # =====================================================
        else:

            if gamma0_plan is None or gamma1_plan is None:
                raise ValueError(
                    "gamma0_plan/gamma1_plan is None, but sampling_info is not a valid dict."
                )

            # 从 γ0 采样条件对 (x0, x1)
            x0, x1, idx_0, idx_1, k_samples = sample_from_ot_plan_sparse(
                gamma0_plan,
                X[t],
                X[t + 1],
                batch_size,
                sampling_info
            )

            values0 = gamma0_plan._values().float()
            values1 = gamma1_plan._values().float()

            gamma0_sample = values0[k_samples].float()
            gamma1_sample = values1[k_samples].float()

            mass0 = torch.ones_like(gamma0_sample).unsqueeze(-1)
            mass1 = (gamma1_sample / (gamma0_sample + 1e-12)).unsqueeze(-1)

            x0 = x0.to(device)
            x1 = x1.to(device)
            mass0 = mass0.to(device)
            mass1 = mass1.to(device)

        # =====================================================
        # 时间采样
        # =====================================================
        delta_t = t_train[t + 1] - t_train[t]

        t_relative = torch.rand(
            x0.shape[0],
            1,
            device=x0.device,
            dtype=x0.dtype
        )

        t_samp = delta_t * t_relative

        # =====================================================
        # 计算 WFR conditional targets
        # =====================================================
        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(
            t_relative,
            delta_t,
            x0,
            x1,
            mass0,
            mass1,
            delta
        )

        ts.append(t_samp[index] + t_train[t])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)

    return (
        torch.cat(ts),
        torch.cat(xts),
        torch.cat(uts),
        torch.cat(gts),
        torch.cat(massts)
    )