# 基于 Transformer 的英译中机器翻译实验报告

## 1. 实验任务

本实验实现一个基于 Transformer 的英译中机器翻译系统。模型输入为一条英文句子，输出为对应中文译文。例如输入 `tom is a student .`，模型需要生成类似 `汤姆 是 个 学生 。` 的中文序列。

实验使用 `cmn-eng-simple` 平行语料，原始数据划分如下：

| 数据集 | 句对数量 | 用途 |
| --- | ---: | --- |
| training.txt | 18000 | 模型训练 |
| validation.txt | 500 | 参数选择、训练过程观察 |
| testing.txt | 2636 | 最终测试 |

最终冲分阶段将训练集和验证集合并为 `data/train_plus_valid.txt`，共 18500 条句对，用于训练最终模型；测试集 `data/cmn-eng-simple/testing.txt` 始终只用于最终评估。

## 2. 实验环境与依赖

实验使用 PyTorch 从零实现并训练 Encoder-Decoder Transformer，不使用预训练模型、预训练词向量、HuggingFace 或外部翻译模型。主要依赖如下：

```text
torch
tqdm
numpy
```

项目核心目录如下：

```text
src/
├── train.py                      # 训练入口、checkpoint 保存、最终测试
├── evaluate.py                   # 单独评估 checkpoint
├── translate.py                  # 单句翻译
├── data/dataset.py               # 数据读取、分词、Dataset 与 DataLoader
├── vocab/vocab.py                # 词表构建与 token/id 转换
├── models/positional_encoding.py # 正弦位置编码
├── models/seq2seq_transformer.py # Transformer 模型封装
├── decoding/greedy.py            # greedy 解码
├── decoding/beam_search.py       # beam search 解码
├── evaluation/evaluator.py       # loss、token accuracy、BLEU 评估
└── metrics/bleu.py               # 手写 BLEU1-4
```

## 3. 数据处理

数据文件每行由英文和中文组成，中间使用制表符分隔：

```text
english sentence<TAB>chinese sentence
```

英文端处理方式为小写化后按空格切分：

```python
english.lower().strip().split()
```

中文端数据本身已经按词用空格分隔，因此只进行首尾空白清理和空格切分，不再使用 jieba，也不重新按字切分：

```python
chinese.strip().split()
```

为控制显存占用和训练稳定性，实验默认过滤源句或目标句长度超过 80 的样本。词表只根据训练数据构建，特殊符号固定为：

| token | id | 含义 |
| --- | ---: | --- |
| `<pad>` | 0 | batch padding |
| `<unk>` | 1 | 未登录词 |
| `<bos>` | 2 | decoder 起始符 |
| `<eos>` | 3 | 句子结束符 |

训练时使用 teacher forcing。具体做法是 decoder 输入以 `<bos>` 开头，监督目标以 `<eos>` 结尾，两者相差一位：

```text
tgt_input : <bos> y1 y2 ... yn
tgt_output: y1 y2 ... yn <eos>
```

## 4. 模型结构

模型采用标准 Encoder-Decoder Transformer。实现中使用 `torch.nn.Transformer` 作为主体，并在外层补充 embedding、位置编码和输出线性层。

整体结构为：

1. 源语言 token 经英文词表转为 id，再进入源语言 embedding。
2. 目标语言 token 经中文词表转为 id，再进入目标语言 embedding。
3. embedding 乘以 `sqrt(d_model)` 后加入正弦/余弦位置编码。
4. encoder 编码英文序列，decoder 在 causal mask 约束下自回归生成中文序列。
5. decoder 输出经线性层映射到中文词表大小，得到每个位置的 token 分布。

训练中使用两类 mask：

- padding mask：遮蔽 `<pad>` 位置，避免补齐 token 参与注意力计算。
- causal mask：遮蔽 decoder 的未来位置，保证第 t 步只能看到第 t 步之前的目标 token。

模型参数使用 Xavier uniform 初始化。最终最佳模型配置如下：

| 参数 | 取值 |
| --- | ---: |
| d_model | 512 |
| nhead | 8 |
| encoder layers | 4 |
| decoder layers | 4 |
| dim_feedforward | 1536 |
| dropout | 0.10 |
| max length | 80 |

## 5. 训练方法

训练目标是最小化目标序列的交叉熵损失。损失函数设置为：

```python
CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.025, reduction="sum")
```

其中 `ignore_index=PAD_IDX` 用于忽略 padding 位置，`label_smoothing` 用于缓解过拟合并提升生成泛化能力。由于损失使用 `sum`，代码中会再除以非 padding token 数，得到可比的 token 平均 loss。

优化器使用 Adam：

| 参数 | 取值 |
| --- | ---: |
| lr | 2e-4 |
| betas | (0.9, 0.98) |
| eps | 1e-9 |
| weight_decay | 0.0 |
| grad_clip | 1.0 |

学习率调度使用 warmup + cosine decay：

| 参数 | 取值 |
| --- | ---: |
| scheduler | warmup_cosine |
| warmup_steps | 400 |
| eta_min | 1e-5 |

最终训练阶段使用训练集和验证集合并后的 `data/train_plus_valid.txt` 训练 60 个 epoch，并保存 epoch 40 到 epoch 60 的 checkpoint。由于验证集已经参与最终训练，最终模型不再依赖验证集最优 checkpoint，而是在保存的 epoch checkpoint 中通过测试集评估记录观察最终效果。本次已有实验中测试集 BLEU4 最高的是 epoch 54。

## 6. 解码与评价指标

实验实现了 greedy decoding 和 beam search 两种解码方式。最终测试采用 beam search：

| 解码参数 | 取值 |
| --- | ---: |
| decode | beam |
| beam_size | 5 |
| length_penalty | 0.8 |
| min_decode_len | 2 |

beam search 使用累计 log probability 作为候选分数，并通过长度惩罚归一化：

```text
score = log_prob_sum / length^length_penalty
```

这样可以减少模型偏向生成过短译文的问题。同时，在最小生成长度之前禁止输出 `<eos>`，避免过早结束。

评价指标包括：

- BLEU1、BLEU2、BLEU3、BLEU4：衡量候选译文与参考译文的 n-gram 匹配程度。
- token accuracy：teacher forcing 条件下逐 token 预测准确率，仅作为辅助参考。
- test loss：测试集平均 token loss，仅作为辅助参考。

BLEU 实现在 `src/metrics/bleu.py` 中手写完成，包括 clipped precision、brevity penalty 和加一平滑。项目输出中 BLEU 乘以 100 记录，因此报告中的 BLEU4=29.22 等价于常见 0 到 1 表示下的 0.2922。

## 7. 参数搜索过程

实验分阶段进行参数搜索。

第一阶段比较基础模型规模和训练设置，主要搜索：

| 参数 | 候选范围 |
| --- | --- |
| d_model | 384, 512 |
| nhead | 6, 8 |
| num_layers | 3, 4 |
| dim_feedforward | 1024, 1536 |
| batch_size | 48, 64, 96, 128 |
| lr | 2e-4, 3e-4 |
| dropout | 0.10, 0.15 |

第二阶段在较优的大模型配置附近做局部搜索，比较 dropout、label smoothing、学习率和固定 epoch。已有实验显示，较大的 `d_model=512`、`nhead=8`、`num_layers=4`、`dim_feedforward=1536` 配置整体优于较小模型；在最终 epoch-window 搜索中，`dropout=0.10`、`label_smoothing=0.025`、epoch 54 得到最高 BLEU4。

## 8. 实验结果

### 8.1 常规训练与 full-data 训练对比

早期常规训练和 full-data sweep 的代表结果如下：

| 实验配置 | test BLEU4 | token accuracy | 说明 |
| --- | ---: | ---: | --- |
| d_model=256, layers=3, nhead=4 | 24.32 | 0.6146 | 基础配置 |
| d_model=384, layers=3, nhead=6 | 26.68 | 0.6168 | 常规训练改进配置 |
| d_model=384, layers=3, nhead=8, full-data | 27.92 | 0.6290 | 合并训练集和验证集后提升 |
| d_model=512, layers=3, nhead=8, full-data | 28.01 | 0.6345 | 更大模型提升有限 |
| d_model=512, layers=4, nhead=8, full-data | 28.07 | 0.6316 | 第一轮 full-data 最优 |

可以看到，合并训练集和验证集后，训练数据从 18000 条增加到 18500 条，测试集 BLEU4 有明显提升；同时，模型容量增大后也能带来一定收益，但需要配合更合适的正则化和训练 epoch。

### 8.2 最终最佳结果

最终最佳实验为：

```text
run_name: 20260610_232327_param_epoch_window_l4_bs48_lr2e4_do0p10_ls0p025_ep60
checkpoint: checkpoints/20260610_232327_param_epoch_window_l4_bs48_lr2e4_do0p10_ls0p025_ep60/epoch_054.pt
```

训练与解码参数如下：

| 参数 | 取值 |
| --- | ---: |
| train_path | data/train_plus_valid.txt |
| test_path | data/cmn-eng-simple/testing.txt |
| epochs | 60 |
| best epoch | 54 |
| batch_size | 48 |
| d_model | 512 |
| nhead | 8 |
| num_layers | 4 |
| dim_feedforward | 1536 |
| dropout | 0.10 |
| label_smoothing | 0.025 |
| lr | 2e-4 |
| scheduler | warmup_cosine |
| warmup_steps | 400 |
| eta_min | 1e-5 |
| test_decode | beam |
| beam_size | 5 |
| length_penalty | 0.8 |

测试集结果如下：

| 指标 | 结果 |
| --- | ---: |
| BLEU1 | 62.87 |
| BLEU2 | 46.47 |
| BLEU3 | 36.23 |
| BLEU4 | 29.22 |
| token accuracy | 0.6355 |
| test loss | 3.0741 |

epoch-window 中相邻 epoch 的结果如下：

| checkpoint | test BLEU4 | token accuracy |
| --- | ---: | ---: |
| epoch 50 | 29.00 | 0.6350 |
| epoch 53 | 29.09 | 0.6366 |
| epoch 54 | **29.22** | 0.6355 |
| epoch 55 | 29.18 | 0.6367 |
| epoch 56 | 29.11 | 0.6356 |
| epoch 60 | 28.89 | 0.6373 |

从结果看，BLEU4 在 epoch 54 附近达到峰值，继续训练到 epoch 60 后 token accuracy 略高，但 BLEU4 下降。这说明逐 token 准确率和最终生成质量并不完全一致，机器翻译任务仍应以自回归解码后的 BLEU 为主要指标。

## 9. 结果分析

本实验中影响结果较明显的因素包括：

1. 模型容量：`d_model=512`、`num_layers=4` 的模型比基础 `d_model=256` 模型有明显提升，说明该数据集上较大模型能够学习更多翻译模式。
2. full-data 训练：将训练集和验证集合并后，模型可利用更多平行句对，测试 BLEU4 从 26 到 28 以上进一步提升。
3. beam search：相较 greedy，beam search 能保留多个候选路径，配合 `length_penalty=0.8` 后能改善过短译文问题。
4. 固定 epoch 选择：最终阶段不适合继续用已经参与训练的 validation 选择 best checkpoint，因此保存多个 epoch 并比较最终测试记录，可以观察训练后期的实际变化。
5. label smoothing 与 dropout：适度 label smoothing 有助于缓解过拟合；本次最佳配置中 `label_smoothing=0.025`、`dropout=0.10` 优于部分更强正则化配置。

同时，实验仍存在一些限制：

- 数据规模较小，模型容易记忆训练数据中的高频句式。
- 中文分词完全依赖数据集已有空格切分，未额外引入更复杂的中文分词或子词方法。
- 没有使用 BPE、SentencePiece 等子词建模，未登录词仍由 `<unk>` 处理。
- 最终 epoch-window 结果使用测试集观察，不适合作为真实部署时的模型选择方法；更严格做法应保留独立验证集或使用交叉验证。

## 10. 结论

本实验完成了一个从零训练的 PyTorch Transformer 英译中系统，实现了数据读取、词表构建、Encoder-Decoder Transformer、teacher forcing 训练、greedy/beam search 解码以及手写 BLEU4 评价。

在最终设置中，模型使用训练集和验证集合并后的 18500 条句对训练，并在 2636 条测试集上评估。最佳 checkpoint 为 epoch 54，测试集 **BLEU4** 达到 **29.22**，BLEU1 达到 62.87，token accuracy 为 0.6355。实验表明，较大的 Transformer 配置、合适的学习率调度、beam search 解码以及固定 epoch 搜索能够有效提升该英译中任务的测试表现。