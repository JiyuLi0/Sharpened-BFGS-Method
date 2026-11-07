# Sharpened BFGS — Python 版本

由本仓库根目录下 `reproduce/` 中的 MATLAB 实验翻译而来，使用 NumPy/SciPy 的 float64 运算。

## 运行

现有 `py312` 环境已经具备依赖。先用模拟数据跑通：

```bash
conda activate xxx
cd /你的仓库路径/Sharpened-BFGS-Method/python
python -m reproduce.main_intergral --demo --iterations 100
```

使用原实验数据：

```bash
python -m reproduce.main_intergral --data /你的数据路径/gisette_scale --iterations 8000
```

数据需自行从 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ 下载。
原仓库没有包含 `gisette_scale`。输入采用 LIBSVM 格式（特征下标从 1 开始），
由 scikit-learn 读取，无需安装 MATLAB 或 LIBSVM 的 MATLAB 接口。
保留原标签映射：0、1 → +1；2、-1 → -1。其他标签会报错。

默认输出 `results/convergence.png` 和 `results/results.npz`，重复运行会覆盖；
可用 `--output results/experiment_name` 指定目录，`--show` 显示图窗。
NPZ 中 `y_gd/y_bfgs/y_greedy/y_sharpened` 为归一化 Newton decrement 历史，
`w_*` 为最终参数，`status_*` 为停止原因。曲线横坐标为迭代次数，纵坐标为
`lambda_t / lambda_0`。

## 与 MATLAB 的对应关系

下表 MATLAB 路径相对于仓库根目录，Python 路径相对于 `python/` 目录。

| MATLAB | Python | 用途 |
| --- | --- | --- |
| `reproduce/main_intergral.m` | `reproduce/main_intergral.py` | 四种算法及绘图 |
| `reproduce/loss_function.m` | `reproduce/loss_function.py` | L2 正则逻辑回归损失 |
| `reproduce/loss_gradient.m` | `reproduce/loss_gradient.py` | 梯度 |
| `reproduce/loss_hessian.m` | `reproduce/loss_hessian.py` | Hessian |
| `reproduce/normalize_row.m` | `reproduce/normalize_row.py` | 按行归一化，保留零行 |
| `reproduce/computeL/compute.py` | `reproduce/computeL/compute.py` | 从 H_0 计算特征值 |

保留原实现的 GD、BFGS、Greedy BFGS、Sharpened BFGS，以及初值
`w_0 = ones(d)/(d*sqrt(d))`。默认 `mu=1e-5`、`L=0.2`、`M=0`、8000 次迭代，
可通过 `--mu`、`--L`、`--M`、`--iterations` 调整。

BFGS 和 Sharpened BFGS 保留原来的
`y = 0.5 * (H_old + H_new) @ s`，没有替换成梯度差。
Greedy BFGS 只做坐标方向更新；Sharpened BFGS 先做步长方向更新，
再做坐标方向更新。缩放因子分别为 `1 + M*r` 和 `(1 + 0.5*M*r)**2`。
没有添加线搜索或改成 L-BFGS。

数值处理有两处调整：使用 `logaddexp/expit` 避免指数溢出；默认在比值
≤ `1e-15` 时停止，也会在浮点精度下参数不再变化时停止，防止收敛后的
BFGS 更新除以零。`--tol 0` 可以关闭容差停止，但仍保留参数不变时的停止。
各算法的历史数组因此可能长度不同。无效曲率会明确报错。

本实现和原脚本一样使用完整的稠密 Hessian，每轮求解线性方程；高维的
gisette 实验可能需要较多内存和时间。模拟数据仅用于验证运行流程，不代表原实验结果。

## 超参数辅助计算

原有 7 个 `.mat` 文件已复制到 `reproduce/computeL/`；它们用于保存矩阵，
不能替代主实验需要的 LIBSVM 训练数据。

```bash
python reproduce/computeL/compute.py
python reproduce/computeL/compute.py reproduce/computeL/a9a.mat
```

这里打印的 μ、L 是所存 `H_0` 的极端特征值，不会自动修改主实验参数，
也不直接证明它们是目标函数的全局曲率界。

## 检查

```bash
python -m unittest discover -s tests -v
```

检查损失的有限差分梯度、梯度的有限差分 Hessian、预处理，以及四种方法
与按原 MATLAB 循环直接书写的参考实现是否一致；不需要额外安装测试框架。
