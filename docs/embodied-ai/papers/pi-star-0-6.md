# π*0.6：从真实经验中学习的 VLA

> π*0.6 用 RECAP 把自主 rollout、人类纠错、稀疏奖励和 advantage-conditioned policy extraction 组合起来，让 VLA 在部署后继续变强。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | π*0.6: a VLA That Learns From Experience (2025) |
| 机构 | Physical Intelligence |
| 关键词 | real-world RL、RECAP、advantage conditioning、human intervention、throughput |
| 定位 | π 系列从 imitation foundation model 走向 self-improving policy |

---

## 1 背景与问题

π0.5 说明泛化来自异构数据，但演示模仿有三个天花板：

1. 人类遥操作速度限制机器人吞吐；
2. 演示中的失误和次优策略会被学进去；
3. 部署中的真实失败模式没有转化为训练信号。

π*0.6 的问题是：如何在真实机器人上稳定地对大型 VLA 做 RL，而不是只做 behavior cloning？


---

## 2 核心贡献

1. 提出 RECAP：RL with Experience and Corrections via Advantage-conditioned Policies；
2. 训练分布型 value function，为观测和任务估计回报分布；
3. 用二值 advantage indicator 条件化 VLA，做稳定的策略提取；
4. 混合 demonstrations、自主 rollout 和人类在线干预；
5. 将 π0.6 后训练为 π*0.6，在咖啡机操作、多样衣物折叠、纸箱装配上显著提升成功率与吞吐。


---

## 3 核心原理与方法详解

RECAP 的核心是先学 value，再用 advantage 条件提取策略。

**1. 分布型价值函数**

将回报离散为 $B$ 个 bin，训练：

$$
\min_\phi
\mathbb{E}_{\tau\in D}
\sum_{o_t\in\tau}
H\left(R_t^{B}(\tau),
p_\phi(V\mid o_t,\ell)\right).
$$

期望价值为：

$$
V^{\pi_{\text{ref}}}(o_t,\ell)
=
\sum_b p_\phi(V=b\mid o_t,\ell)v(b).
$$

**2. Advantage 指示**

定义：

$$
I_t=
\mathbf{1}\left(
A^{\pi_{\text{ref}}}(o_t,a_t,\ell)>\epsilon_\ell
\right).
$$

人类干预动作强制 $I_t=1$。

**3. 条件策略提取**

$$
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
-\log \pi_\theta(a_t\mid o_t,\ell)
-\alpha \log \pi_\theta(a_t\mid I_t,o_t,\ell)
\right].
$$

推理时设置高优势条件，使模型偏向数据中更有利于任务成功的动作模式。


---

## 4 数据与训练

- 基础模型为 π0.6：π0.5 架构的演进，保留 Flow Matching 动作专家与文本推理能力；
- 使用 Knowledge Insulation 式训练：文本/FAST 动作与连续动作专家并行，动作专家对主干 stop-gradient，避免相互干扰；
- value function 与 VLA 类似，但使用较小 VLM 初始化；
- 预训练数据为数十万小时级演示，覆盖大量任务与机器人；
- 部署后每轮执行：自主采样 → 人工奖励标注 → 可选 teleoperation 纠错 → value 更新 → policy 更新；
- 任务包括浓缩咖啡、衣物折叠、纸箱折叠与贴标。


---

## 5 实验、结果与解读

论文用 throughput（每小时成功任务数）和 success rate 评估：

- 最难任务上 throughput 提升超过 2 倍，失败率约减半；
- 除多样衣物外，多数最终任务成功率进入 90%+ 区间；
- 纸箱装配各子阶段成功率均提升，多数失败来自超时；
- 与 AWR、PPO 相比，RECAP 的策略提取在高吞吐任务上更稳定；
- 针对严格衣领方向的折叠任务，两轮 RECAP 后成功率可达约 97%；
- RECAP 能移除特定失败模式，而不需要重新采集演示。


---

## 6 精读要点与实践建议

- **RL 不只优化成功率，也优化速度和吞吐**，这是部署可用性的关键；
- **人类干预不是噪声，而是高价值正样本**；
- **advantage conditioning 比 policy gradient 更适合生成式动作头**，避免对 Flow Matching 采样过程求高方差梯度；
- **数据闭环**：自主失败、人类纠错和奖励标注共同构成部署后训练集；
- **实践建议**：真机 RL 应从可靠预训练模型开始，否则探索成本和安全风险不可控。


---

## 7 局限与后续影响

- 依赖人工奖励标注、干预和 reset，还不是完全自主 RL；
- value function 的误差会影响策略提取；
- 任务专用后训练可能牺牲部分通用性，论文称最终通用模型另需重新训练；
- 长尾奖励判断仍困难；
- 后续 π0.7 的目标之一是把 specialist 能力蒸馏回可组合的通用模型，并增强 steerability。


---

## 相关阅读

- [π0.5 论文精读](./pi-0-5.md)
- [π0.7 论文精读](./pi-0-7.md)
- [强化学习与对齐](../../training/reinforcement-learning.md)

---

## 参考文献

1. Physical Intelligence. *π*0.6: a VLA That Learns From Experience*, 2025. [项目页面](https://www.pi.website/blog/pistar06)
2. Black et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025.
3. Physical Intelligence. *VLAs that Train Fast, Run Fast, and Generalize Better*, 2025.
4. Levine et al. *Offline Reinforcement Learning: Tutorial, Review, and Perspectives*, 2020.

