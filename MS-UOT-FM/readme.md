# MS-UOT-FM Release

主要修改：
1. utils.py 后面新加相应函数
2. train.py 后面新加相应函数

主要问题：
1. 稀疏采样有上限（sample_from_ot_plan_sparse 函数）
2. 难以稀疏计算 mass0，mass1（get_batch_sparse 函数）
3. 实现最细尺度随机匹配（似乎能天然地解决前两个问题）
