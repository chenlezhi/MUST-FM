# MS-UOT-FM Release

主要修改：

4.17 更新：
1. utils.py 后面新加相应函数
2. train.py 后面新加相应函数

4.20 更新：
1. 实现根据数据尺度自适应进行多尺度 OT 计算
2. 实现完全稀疏的 mass 计算


主要问题：
1. 稀疏采样有上限（sample_from_ot_plan_sparse 函数）：torch.multinomial(probs, batch_size, replacement=True) 函数本身限制 probs 元素数量
2. 实现最细尺度随机匹配（似乎能天然地解决前两个问题）
