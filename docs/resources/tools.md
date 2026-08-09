# 深度学习常用工具

> 开发、训练、调试与部署的常用工具及其核心命令速查。

---

## 目录

1. [Git 版本控制](#1-git)
2. [Conda 虚拟环境](#2-conda)
3. [pip 包管理](#3-pip)
4. [Linux 常用命令](#4-linux)
5. [Tmux 终端复用](#5-tmux)
6. [Vim 编辑器](#6-vim)
7. [常用 Python 库](#7-python)
8. [Docker](#8-docker)
9. [监控与调试](#9)

---

## 1. Git 版本控制

### 基本操作

```bash
git init                          # 初始化仓库
git clone <url>                   # 克隆仓库
git add <file>                    # 暂存文件
git commit -m "message"           # 提交
git push origin main              # 推送到远程
git pull origin main              # 拉取远程更新
```

### 分支操作

```bash
git branch                        # 查看分支
git checkout -b <branch>          # 创建并切换分支
git merge <branch>                # 合并分支
git branch -d <branch>            # 删除分支
```

### 撤销操作

```bash
git checkout -- <file>            # 撤销工作区修改
git reset HEAD <file>             # 撤销暂存
git reset --soft HEAD~1           # 撤销最近一次 commit（保留修改）
git reset --hard HEAD~1           # 完全撤销最近一次 commit
```

### 日志与差异

```bash
git log --oneline --graph         # 简洁日志
git diff                          # 查看工作区差异
git diff --staged                 # 查看暂存区差异
git stash                         # 暂存当前修改
git stash pop                     # 恢复暂存
```

---

## 2. Conda 虚拟环境

```bash
# 创建环境
conda create -n myenv python=3.10

# 激活/退出环境
conda activate myenv
conda deactivate

# 列出所有环境
conda env list

# 安装包
conda install numpy pandas
conda install -c conda-forge <package>

# 导出/导入环境
conda env export > environment.yml
conda env create -f environment.yml

# 删除环境
conda env remove -n myenv
```

---

## 3. pip 包管理

```bash
# 安装包
pip install <package>==<version>
pip install -r requirements.txt

# 卸载
pip uninstall <package>

# 列出已安装包
pip list
pip freeze > requirements.txt

# 查看包信息
pip show <package>

# 使用国内镜像源
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4. Linux 常用命令

### 文件操作

```bash
ls -la                           # 列出文件（详细信息）
cd <dir>                         # 切换目录
pwd                              # 显示当前路径
mkdir -p <dir>                   # 创建目录
rm -rf <dir>                     # 删除目录（谨慎！）
cp -r <src> <dst>                # 复制
mv <src> <dst>                   # 移动/重命名
find . -name "*.py"              # 查找文件
```

### 文本处理

```bash
grep "pattern" <file>            # 搜索文本
grep -r "pattern" .              # 递归搜索
wc -l <file>                     # 统计行数
head -n 10 <file>                # 查看前 10 行
tail -f <file>                   # 实时查看文件末尾
cat <file> | sort | uniq -c      # 排序去重统计
```

### 系统监控

```bash
htop                             # 进程监控
nvidia-smi                       # GPU 监控
nvidia-smi -l 1                  # 每秒刷新 GPU 状态
df -h                            # 磁盘使用
du -sh <dir>                     # 目录大小
free -h                          # 内存使用
```

### 进程管理

```bash
ps aux | grep python             # 查找进程
kill -9 <PID>                    # 终止进程
nohup <command> &                # 后台运行
jobs                             # 查看后台任务
fg                              # 调回前台
```

### 文件传输

```bash
scp <file> user@host:<path>      # 上传文件
scp -r <dir> user@host:<path>    # 上传目录
scp user@host:<path> <local>     # 下载文件
rsync -avz <src> <dst>           # 同步文件（支持断点续传）
```

---

## 5. Tmux 终端复用

```bash
# 会话管理
tmux new -s <name>               # 创建命名会话
tmux ls                          # 列出会话
tmux attach -t <name>            # 连接会话
tmux kill-session -t <name>      # 关闭会话

# 窗口操作 (Ctrl+B 后)
c                                # 创建窗口
&                                # 关闭窗口
n / p                            # 切换窗口
,                                # 重命名窗口

# 面板操作 (Ctrl+B 后)
%                                # 垂直分割
"                                # 水平分割
方向键                           # 切换面板
Ctrl+方向键                      # 调整面板大小
x                                # 关闭面板
```

---

## 6. Vim 编辑器

### 基本操作

```bash
i                                # 进入插入模式
Esc                              # 退出插入模式
:w                               # 保存
:q                               # 退出
:wq                              # 保存并退出
:q!                              # 强制退出（不保存）
```

### 导航

```bash
h / j / k / l                    # 左 / 下 / 上 / 右
0 / $                            # 行首 / 行尾
gg / G                           # 文件开头 / 文件结尾
/<pattern>                       # 搜索
n / N                            # 下一个 / 上一个匹配
```

### 编辑

```bash
dd                               # 删除整行
yy                               # 复制整行
p                                # 粘贴
u                                # 撤销
Ctrl+r                           # 重做
```

---

## 7. 常用 Python 库

### 数据处理

| 库 | 用途 |
|----|------|
| NumPy | 数值计算 |
| Pandas | 表格数据处理 |
| SciPy | 科学计算 |
| Polars | 高性能 DataFrame |

### 可视化

| 库 | 用途 |
|----|------|
| Matplotlib | 基础绘图 |
| Seaborn | 统计可视化 |
| Plotly | 交互式图表 |
| TensorBoard | 训练可视化 |
| WandB | 实验跟踪 |

### 深度学习

| 库 | 用途 |
|----|------|
| PyTorch | 深度学习框架 |
| Transformers | 预训练模型 |
| Datasets | 数据集处理 |
| Accelerate | 分布式训练 |
| PEFT | 参数高效微调 |
| Diffusers | 扩散模型 |
| Timm | 预训练视觉模型 |

---

## 8. Docker

```bash
# 镜像管理
docker images                     # 列出镜像
docker pull <image>               # 拉取镜像
docker rmi <image>                # 删除镜像

# 容器管理
docker run --gpus all -it <image> # 启动 GPU 容器
docker run -v $(pwd):/workspace <image>  # 挂载目录
docker ps                         # 查看运行中的容器
docker ps -a                      # 查看所有容器
docker stop <container>           # 停止容器
docker rm <container>             # 删除容器

# 进入容器
docker exec -it <container> bash
```

---

## 9. 监控与调试

### GPU 监控

```bash
nvidia-smi                        # 基本信息
nvidia-smi -l 1                   # 每秒刷新
nvidia-smi dmon                   # 设备监控
gpustat                           # 简洁 GPU 状态（需 pip install gpustat）
```

### 训练监控

```bash
# TensorBoard
tensorboard --logdir=./logs --port=6006

# WandB
wandb login
wandb init
```

### 性能分析

```python
# PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(input)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```
