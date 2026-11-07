# Sharpened BFGS Method

本仓库包含 Sharpened BFGS 的 MATLAB 实验代码和对应的 Python 版本，
对比 GD、BFGS、Greedy BFGS 和 Sharpened BFGS。

- [MATLAB 版本](reproduce/README.md)：代码位于 `reproduce/`。
- [Python 版本与使用说明](python/README.md)：代码位于 `python/`。

## Python 快速运行

在已安装 NumPy、SciPy、Matplotlib 和 scikit-learn 的 Python 环境中，
从仓库根目录执行：

```bash
cd python
python -m reproduce.main_intergral --demo --iterations 100
```

依赖列表见 `python/requirements.txt`。模拟数据用于验证运行流程；
原实验需要的 LIBSVM 数据请按各版本 README 的说明另外下载。
