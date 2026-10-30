<div align="center">

# Retrieving from Your Own Mistakes

### 从自己的错误中检索：面向多跳检索增强生成的嵌入模型 On-Policy 训练

[English](README.md) · **简体中文**

[![arXiv](https://img.shields.io/badge/arXiv-2610.40409-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.40409)
![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![Encoder](https://img.shields.io/badge/encoder-Qwen3--Embedding--0.6B-6E4FC9)
![LLM calls](https://img.shields.io/badge/LLM%20calls%20at%20retrieval-0-2E8B57)


📄 **论文：** [arXiv:2610.40409](https://arxiv.org/abs/2610.40409)

</div>

<p align="center">
  <img src="assets/overview.png" width="96%" alt="On-policy state training 概览">
</p>

> **(a)** Teacher forcing 只在 gold 历史上训练链式检索器，测试时检索器自己犯的错误会一直留在查询里。
> **(b)** On-policy state training 用当前编码器做 roll-in，把每个访问到的状态标注为最早缺失的 gold 段落，并在 gold 状态和访问状态的混合上训练，访问状态每个 epoch 刷新一次。

---

## 亮点

- **不调用语言模型也很强。** 0.6B 的链式嵌入检索器只训练查询侧 LoRA、只用一个稠密索引，在 MuSiQue 上完整取回证据链的比例高于基于图的 RAG 系统和 7B 的搜索 agent。
- **状态失配很严重。** 到第三步时，teacher-forced 链式检索器访问的状态里约一半或更多已经包含它自己检索错的段落，仅这些错误段落就让它下一步的成功率下降 11 个点。
- **只改一处，收益明确。** 只改变训练状态的分布，MuSiQue 上的完全覆盖率就从 44.6 提升到 50.5（同编码器、同配方的 teacher forcing 对比），而且不经额外训练就能迁移到更长的链。
- **收益出现在失配所预测的位置。** 在同一组固定状态上，on-policy 编码器在出错之后恢复的频率约为两倍；合成污染解释不了的那部分收益，来自用当前编码器刷新访问状态。

## 为什么重要

<table>
<tr>
<td width="50%"><img src="assets/diagnosis.png" alt="错误累积与同状态恢复率"></td>
<td width="50%">

**(a) 错误会累积。** 在四跳 MuSiQue 问题上，teacher-forced 检索器第 2 到 4 步的状态中，分别有 19%、70% 和 89% 已经包含错误段落。

**(b) 同一组状态上的恢复。** 在同一组固定状态上打分，历史干净时各编码器表现接近；历史里有两个及以上错误段落时，on-policy state training 的成功率约为 teacher forcing 的两倍。

</td>
</tr>
</table>

## 一段话讲清方法

链式检索器把状态 `(问题, 已检索的段落)` 编码成查询，检索下一个段落。每个 epoch，当前编码器在训练问题上跑一遍 greedy 检索链。每个仍缺 gold 段落的访问状态都被标注为**最早**缺失的 gold 段落，其余缺失的 gold 段落从对比学习的分母中**屏蔽**，因此任何正确的后续都不会被惩罚。随后在这些访问状态和 teacher-forced 状态各占一半的混合上训练查询编码器。模型结构、损失、冻结编码器给出的负例和文档索引都与 teacher forcing 相同，两种训练方式只在训练状态上不同。

## 实验结果

MuSiQue 和我们构建的两个长链 benchmark 上的检索结果（我们的检索器只在 MuSiQue 上训练，结果为五个种子的均值）。

| 方法 | MuSiQue R@5 | MuSiQue C@5 | HotpotQA-Long C@10 | 2Wiki-Long C@10 | 检索时调用 LLM |
|---|:---:|:---:|:---:|:---:|:---:|
| Dense（Qwen3-Embedding-0.6B） | 55.1 | 18.9 | 1.0 | 2.0 | – |
| HippoRAG 2 | 58.2 | 23.1 | 1.3† | 2.3† | ✓ |
| MDR | 58.3 | 31.9 | 14.7 | 14.3 | – |
| Q-RAG | 61.7 | 36.2 | 5.7 | 2.3 | – |
| Search-R1（7B） | 66.7 | 36.5 | 25.7 | **37.0** | ✓ |
| **On-policy state training** | **73.5** | **49.3** | **29.6** | 32.8 | – |

† 在缩减语料上运行，比完整语料的设定更简单。完整的 @5 与 @10 召回率和覆盖率见论文 Table 1。

<p align="center">
  <img src="assets/bytype.png" width="96%" alt="按题型的结果">
</p>
<p align="center"><sub>按题型划分的召回率与完全覆盖率。</sub></p>

消融实验（同一个种子）：

| | MuSiQue C@5 | HotpotQA-Long C@10 | 2Wiki-Long C@10 |
|---|:---:|:---:|:---:|
| **On-policy state training** | **50.5** | **28.0** | **34.7** |
| 去掉刷新 | 48.4 | 22.7 | 28.3 |
| 去掉访问状态（即 teacher forcing） | 44.6 | 22.3 | 24.3 |
| 去掉链式检索 | 18.9 | 1.0 | 1.7 |

## 查询指向哪里

<p align="center">
  <img src="assets/trajectories.png" width="96%" alt="查询轨迹">
</p>

Teacher-forced 的检索链一旦取到非 gold 段落，下一步的查询就指向非 gold 段落并停留在那里。On-policy 的查询沿着 gold 行逐跳前进；即使从错误段落附近出发，也会在一两步内回到 gold 行。

<p align="center">
  <img src="assets/flow.png" width="80%" alt="逐步结果">
</p>
<p align="center"><sub>三跳、四跳 MuSiQue 问题的逐步结果；深色条带表示从未命中恢复到命中。</sub></p>

## 仓库结构

```
on-policy-state-training/
├── src/
│   ├── core.py, states.py, train.py     编码器与状态文本、各训练条件的训练状态、训练
│   ├── inference.py, evaluate.py        greedy 链式检索与 MuSiQue 评测
│   ├── initial_mix.py, rollin_fixed.py  第一个 epoch 的混合状态、只收集一次的访问状态
│   ├── static_reference.py              合成污染的匹配统计
│   ├── shared_pool.py, overlap.py       共享负例池
│   ├── probe.py, metrics.py             固定状态恢复率探针、R@k 与 C@k
│   ├── longchain/                       HotpotQA-Long 与 2Wiki-Long 上的零样本评测（T = 8）
│   └── run_paper.sh                     按依赖顺序列出全部命令
├── experiments/
│   ├── beam/          greedy 与 beam 解码
│   ├── qa/            固定阅读器的下游问答
│   ├── reduced_corpus/
│   └── baselines/     Search-R1、Q-RAG、MDR、G-Retriever、LinearRAG
├── benchmarks/
│   ├── code/          构建流程、prompt、验证与捷径过滤
│   └── data/          发布的问题与有序 gold 链
├── figures/           绘图脚本
└── assets/            本 README 用到的图片
```

| 目录 | 对应论文 |
|---|---|
| `src` | 第 3–5 节，Table 1、3、4、11，附录 A–C |
| `experiments/beam` | 附录 F.3 |
| `experiments/qa` | Table 2，附录 F.1 |
| `experiments/reduced_corpus` | 附录 D |
| `experiments/baselines` | Table 1 |
| `benchmarks` | 第 5.1 节，附录 E |
| `figures` | Figure 2–7 |

每个目录都有自己的 README；`src/README.md` 列出了每个训练条件对应论文里的哪一行。

## 快速开始

```bash
pip install -r requirements.txt
cd src

python train.py --condition RN         --lr 3e-4 --seed 13   # teacher forcing
python initial_mix.py                                        # 用 teacher-forced 编码器的 roll-in 构建第一个 epoch 的混合状态
python train.py --condition OP-refresh --lr 3e-4 --seed 13   # on-policy state training

python evaluate.py --condition OP-refresh --seed 13 --split design                 # MuSiQue 测试题
python longchain/eval54.py --dataset hotpotqa_long --condition OP-refresh --seed 13  # 零样本迁移
python longchain/eval54.py --dataset 2wiki_long    --condition OP-refresh --seed 13
```

`run_paper.sh` 还包括消融（`Fixed-OP`、`SQ-LoRA-v2`）、Table 4 的对照（`Static-corrupt`、`Static-refresh`、`Pool-resample`）、混合比例（`Mix-25`、`Mix-75`）、恢复率探针和共享负例池。

## Benchmark

| | 链 | 测试题 | 语料 |
|---|---|---|---|
| **HotpotQA-Long** | 沿维基百科超链接的 3、4、5 跳 | 每种长度 100 题 | 554,812 段 |
| **2Wiki-Long** | 沿 Wikidata 函数关系的 3–5 跳，外加 3–4 个实体的比较题 | 170 / 45 / 12 条链，73 道比较题 | 220,000 段 |

问题里只出现起始实体。另一家族的模型复核可回答性和答案唯一性；仅凭问题就能被 BM25 检索到最后一个段落的题目会被删除。`benchmarks/data` 中是带有序 gold 链的测试题。语料超过 GitHub 的单文件上限，需要用 `benchmarks/code` 重新构建（见该目录 README）。

## 环境与说明

- Python 3.11、torch 2.5.1（CUDA 12.4）、transformers 4.51.3、peft 0.17.1、numpy 1.26.4、faiss 1.9.0。baseline 使用各自的官方仓库。
- **不包含：** 模型权重、LoRA adapter、段落向量、完整的长链语料、日志和中间结果。
- 数据路径由 `OPST_ROOT` 指定（见 `src/README.md`），`R19_MODEL` 指向本地的 Qwen3-Embedding-0.6B，baseline 脚本从 `WORK` 读取外部仓库。
- 调用语言模型的脚本从环境变量读取 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`。

## 引用

```bibtex
@article{qiu2026retrieving,
  title         = {Retrieving from Your Own Mistakes: On-Policy Training of Embedding Models
                   for Multi-Hop Retrieval-Augmented Generation},
  author        = {Qiu, Chenjie and Zeng, Zixuan and Shi, Shaobin and Chen, Bingfeng and Luo, Dongsheng and Xu, Boyan},
  journal       = {arXiv preprint arXiv:2610.40409},
  eprint        = {2610.40409},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.40409}
}
```
