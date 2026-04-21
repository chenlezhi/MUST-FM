__all__ = ["pretrain", "train", "multiscale_train"]

import logging

import numpy as np
import torch
from tqdm import tqdm

from utils import compute_uot_plans, get_batch


def pretrain(
    model,              # FNet实例，包含v_net和g_net
    df,                 # 原始DataFrame，含'samples'时间列和特征列
    optimizer_1,        # v_net的优化器（如Adam）
    optimizer_2,        # g_net的优化器（可独立设置不同lr）
    scheduler_1=None,   # v_net的学习率调度器（可选）
    scheduler_2=None,   # g_net的学习率调度器（可选）
    n_epoch=1000,       # 训练总轮数
    test_interval=100,  # 评估间隔（当前代码未使用，预留接口）
    batch_size=256,     # 每个mini-batch的采样点数
    hold_one_out=False, # 是否留一法评估（留出一个时间点不训练）
    hold_out="random",  # 留出的时间点标识
    logger=None,        # 日志对象（用于记录训练进度）
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 计算设备
    save_dir=None,      # 模型保存路径（预留）
    relative_mass=None, # 各时间点相对质量参考（用于质量正则）
    delta=1.0,          # 【关键】WFR度量中的增长惩罚系数δ
    use_mini_batch=False,  # 是否使用mini-batch近似WFR-OET耦合
    chunk_size=2000,    # mini-batch OET的chunk大小
    ema=None,           # 指数移动平均（可选，用于稳定训练）
):

    """
    数据预处理
    """
    time_labels = df["samples"].to_numpy()  # 提取时间列 [0,0,1,1,2,2,...]
    data = df.iloc[:, 1:].to_numpy()  # 提取特征列（去掉 samples ）
    x_all = [data[time_labels == t] for t in np.unique(time_labels)]  # 按时间点分组
    x_selected = [data[time_labels == v] for v in np.unique(time_labels) if v != hold_out]  # 留一法处理 hold_out
    t_train = [v for v in np.unique(time_labels) if v != hold_out]

    if logger is not None:
        logger.info("Begin flow and growth matching")

    """
    计算 WFR-OET （此处修改为我们的 Multiscale-OT）
    """
    gamma0_plans, gamma1_plans, sampling_info = compute_uot_plans(
        x_selected,
        t_train,
        delta=delta,
        draw=False,
        use_mini_batch_uot=use_mini_batch,
        chunk_size=chunk_size,
    )

    progress_bar = tqdm(range(n_epoch), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []

    """
    开始 FM 训练
    """
    for epoch in progress_bar:  # 设置两个 optimizer，允许为 v 和 g 设置不同学习率（实践中均为 0.005）
        optimizer_1.zero_grad()  # 清空 v_net 梯度
        optimizer_2.zero_grad()  # 清空 g_net 梯度

        """
        1. 随机选择时间对 (tk, tk+1) ~ Uniform
        2. 从半耦合γ0采样条件变量: (x_tk, x_tk+1) ~ γ0
        3. 计算traveling Dirac参数 (公式3.6):
           τ = tan(||x_tk+1 - x_tk||/(2δ)),  A, B, ω0, l
        4. 计算质量演化 m_t(x_tk, x_tk+1) (公式3.5):
           m(t) = A*t² - 2B*t + m0,  m0=1, m1=γ1/γ0
        5. 采样中间状态 (公式4.8):
           η_t = x_tk + ω0 * Λ_t,  x ~ N(η_t, σ²I)
        6. 计算目标场 (公式3.1, 3.5):
           u_t = ω0 / m_t,   g_t = d/dt ln(m_t)
        7. 返回: (时间戳t, 状态x, 目标速度u, 目标增长率g, 质量权重m_t)
        """
        t, xt, ut, gt, masst = get_batch(
            x_selected,
            t_train,
            batch_size,
            gamma0_plans,
            gamma1_plans,
            delta,
            relative_mass,
            sampling_info,
        )

        # 前向传播
        vt = model.v_net(t, xt)
        gt_pred = model.g_net(t, xt)

        """
        CUFM loss
        """
        vloss = torch.mean((vt - ut) ** 2 * masst)
        gloss = torch.mean((gt_pred - gt) ** 2 * masst)
        loss = vloss + gloss

        vloss_list.append(vloss.item())
        gloss_list.append(gloss.item())
        loss_list.append(loss.item())

        loss.backward()

        optimizer_1.step()
        optimizer_2.step()

        if scheduler_1 is not None:
            scheduler_1.step()
        if scheduler_2 is not None:
            scheduler_2.step()
        if ema is not None:
            ema.update()

        # 日志
        if logger is not None:
            logger.info(
                "Epoch %d: loss=%.6f, vloss=%.6f, gloss=%.6f",
                epoch,
                loss.item(),
                vloss.item(),
                gloss.item(),
            )
        else:
            logging.info(
                "Epoch %d: loss=%.6f, vloss=%.6f, gloss=%.6f",
                epoch,
                loss.item(),
                vloss.item(),
                gloss.item(),
            )
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.6f}",
                "vloss": f"{vloss.item():.6f}",
                "gloss": f"{gloss.item():.6f}",
            }
        )

    return model, vloss_list, gloss_list, loss_list


def train(
    model,
    df,
    n_epoch=1000,
    test_interval=100,
    batch_size=256,
    hold_one_out=False,
    hold_out="random",
    logger=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir=None,
    relative_mass=None,
    delta=1.0,
    use_mini_batch=False,
    chunk_size=2000,
    lr_v=0.005,
    lr_g=0.005,
    eta_min=1e-5,
    ema=None,
):
    optimizer_1 = torch.optim.Adam(model.v_net.parameters(), lr=lr_v)
    optimizer_2 = torch.optim.Adam(model.g_net.parameters(), lr=lr_g)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=n_epoch, eta_min=eta_min)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=n_epoch, eta_min=eta_min)

    return pretrain(
        model,
        df,
        optimizer_1,
        optimizer_2,
        scheduler_1=scheduler_1,
        scheduler_2=scheduler_2,
        n_epoch=n_epoch,
        test_interval=test_interval,
        batch_size=batch_size,
        hold_one_out=hold_one_out,
        hold_out=hold_out,
        logger=logger,
        device=device,
        save_dir=save_dir,
        relative_mass=relative_mass,
        delta=delta,
        use_mini_batch=use_mini_batch,
        chunk_size=chunk_size,
        ema=ema,
    )


###############################################################################################
"""
CLZ:
以下为新增内容
"""
###############################################################################################

from utils import compute_multiscale_uot_plans, get_data, get_batch_sparse

def multiscale_train(
    model,              # FNet实例，包含v_net和g_net
    df,                 # 原始DataFrame，含'samples'时间列和特征列与scale标注
    optimizer_1,        # v_net的优化器（如Adam）
    optimizer_2,        # g_net的优化器（可独立设置不同lr）
    scheduler_1=None,   # v_net的学习率调度器（可选）
    scheduler_2=None,   # g_net的学习率调度器（可选）
    n_epoch=1000,       # 训练总轮数
    test_interval=100,  # 评估间隔（当前代码未使用，预留接口）
    batch_size=256,     # 每个mini-batch的采样点数
    hold_one_out=False, # 是否留一法评估（留出一个时间点不训练）
    hold_out="random",  # 留出的时间点标识
    logger=None,        # 日志对象（用于记录训练进度）
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),  # 计算设备
    save_dir=None,      # 模型保存路径（预留）
    relative_mass=None, # 各时间点相对质量参考（用于质量正则）
    delta=1.0,          # 【关键】WFR度量中的增长惩罚系数δ
    use_mini_batch=False,  # 是否使用mini-batch近似WFR-OET耦合
    chunk_size=2000,    # mini-batch OET的chunk大小
    ema=None,           # 指数移动平均（可选，用于稳定训练）
):

    """
    数据预处理
    """
    df_tmp = get_data(df)

    time_labels = df_tmp["samples"].to_numpy()  # 提取时间列 [0,0,1,1,2,2,...]
    data = df_tmp.iloc[:, 1:].to_numpy()  # 提取特征列（去掉 samples ）

    x_all = [data[time_labels == t] for t in np.unique(time_labels)]  # 按时间点分组
    x_selected = [data[time_labels == v] for v in np.unique(time_labels) if v != hold_out]  # 留一法处理 hold_out
    t_train = [v for v in np.unique(time_labels) if v != hold_out]

    if logger is not None:
        logger.info("Begin flow and growth matching")

    """
    计算 WFR-OET (Multiscale-OT)
    """
    gamma0_plans, gamma1_plans, sampling_info = compute_multiscale_uot_plans(
        df,
        x_selected,
        t_train,
        delta=delta,
        use_mini_batch_uot=use_mini_batch,  # 应为 False
        chunk_size=chunk_size,
    )

    progress_bar = tqdm(range(n_epoch), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []

    """
    开始 FM 训练
    """
    for epoch in progress_bar:  # 设置两个 optimizer，允许为 v 和 g 设置不同学习率（实践中均为 0.005）
        optimizer_1.zero_grad()  # 清空 v_net 梯度
        optimizer_2.zero_grad()  # 清空 g_net 梯度

        """
        1. 随机选择时间对 (tk, tk+1) ~ Uniform
        2. 从半耦合γ0采样条件变量: (x_tk, x_tk+1) ~ γ0
        3. 计算traveling Dirac参数 (公式3.6):
           τ = tan(||x_tk+1 - x_tk||/(2δ)),  A, B, ω0, l
        4. 计算质量演化 m_t(x_tk, x_tk+1) (公式3.5):
           m(t) = A*t² - 2B*t + m0,  m0=1, m1=γ1/γ0
        5. 采样中间状态 (公式4.8):
           η_t = x_tk + ω0 * Λ_t,  x ~ N(η_t, σ²I)
        6. 计算目标场 (公式3.1, 3.5):
           u_t = ω0 / m_t,   g_t = d/dt ln(m_t)
        7. 返回: (时间戳t, 状态x, 目标速度u, 目标增长率g, 质量权重m_t)
        """
        t, xt, ut, gt, masst = get_batch_sparse(
            x_selected,
            t_train,
            batch_size,
            gamma0_plans,
            gamma1_plans,
            delta,
            relative_mass,
            sampling_info                
        )

        # 前向传播
        vt = model.v_net(t, xt)
        gt_pred = model.g_net(t, xt)

        """
        CUFM loss
        """
        vloss = torch.mean((vt - ut) ** 2 * masst)
        gloss = torch.mean((gt_pred - gt) ** 2 * masst)
        loss = vloss + gloss

        vloss_list.append(vloss.item())
        gloss_list.append(gloss.item())
        loss_list.append(loss.item())

        loss.backward()

        optimizer_1.step()
        optimizer_2.step()

        if scheduler_1 is not None:
            scheduler_1.step()
        if scheduler_2 is not None:
            scheduler_2.step()
        if ema is not None:
            ema.update()

        # 日志
        if logger is not None:
            logger.info(
                "Epoch %d: loss=%.6f, vloss=%.6f, gloss=%.6f",
                epoch,
                loss.item(),
                vloss.item(),
                gloss.item(),
            )
        else:
            logging.info(
                "Epoch %d: loss=%.6f, vloss=%.6f, gloss=%.6f",
                epoch,
                loss.item(),
                vloss.item(),
                gloss.item(),
            )
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.6f}",
                "vloss": f"{vloss.item():.6f}",
                "gloss": f"{gloss.item():.6f}",
            }
        )

    return model, vloss_list, gloss_list, loss_list