# 基于 Transformer 的英中机器翻译

本项目用于完成 `docs/question.md` 中的课程作业：使用 Transformer 将英文句子翻译为中文。

模型使用 PyTorch 编写，并自行实现标准 Transformer 结构，没有调用 `torch.nn.Transformer`。

## 项目内容

核心功能包括：

- 英文/中文词表加载
- 平行语料 Dataset 和 batch padding
- 正弦位置编码
- 多头注意力机制
- Encoder 和 Decoder 堆叠
- Decoder 因果 mask
- 源序列 padding mask
- 前馈网络、残差连接和 LayerNorm
- greedy 解码
- beam search 解码
- BLEU 评估
- tqdm 训练进度条
- label smoothing、Noam warmup 学习率调度和 early stopping

主要文件：

```text
src/train.py              训练入口
src/eval.py               测试/验证集评估入口
src/translate.py          单句翻译入口
src/data/dataset.py       数据读取、词表加载、batch 组装
src/models/transformer.py 自定义 Transformer 模型
src/utils/constants.py    常量定义
src/utils/metrics.py      BLEU 指标实现
scripts/train.sh          训练脚本
scripts/evaluate.sh       评估脚本
scripts/translate.sh      单句翻译脚本
scripts/run_experiment.sh 一键训练、评估和报告生成脚本
tests/smoke_test.py       基础链路测试
```

## 数据集

数据位于 `data/cmn-eng-simple`。

| 数据划分 | 句子数 |
| --- | ---: |
| 训练集 | 18000 |
| 验证集 | 500 |
| 测试集 | 2636 |

每一行格式为：

```text
英文分词结果<TAB>中文分词结果
```

词表文件已经给出：

```text
word2int_en.json
int2word_en.json
word2int_cn.json
int2word_cn.json
```

特殊 token 编号固定为：

```text
<PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3
```

## 环境

推荐使用已经配置好的 conda 环境：

```bash
conda activate translation-transformer
```

如果当前 shell 没有初始化 conda，也可以直接使用环境中的 Python：

```bash
~/miniconda3/envs/translation-transformer/bin/python
```

当前验证过的环境：

```text
Python 3.12.13
PyTorch 2.12.0+cu130
CUDA 13.0
NumPy 2.4.6
```

如果需要重新安装依赖，可以使用 NJU PyPI 镜像：

```bash
python -m pip install -r requirements.txt \
  -i https://mirror.nju.edu.cn/pypi/web/simple \
  --trusted-host mirror.nju.edu.cn
```

## 基础测试

运行 smoke test，检查数据加载、模型前向传播和 greedy 解码是否正常：

```bash
python -m tests.smoke_test
```

成功时输出：

```text
smoke test passed
```

## 训练

默认配置是一个适合本数据集规模的小型 Transformer：

```bash
./scripts/train.sh
```

脚本会优先使用 `~/miniconda3/envs/translation-transformer/bin/python`，也可以通过 `PYTHON` 环境变量指定解释器：

```bash
PYTHON=python ./scripts/train.sh
```

如果需要覆盖默认训练参数，可以把参数追加到脚本后面：

```bash
./scripts/train.sh --epochs 5 --batch-size 128
```

训练过程中会输出训练损失、验证损失、困惑度和验证集 BLEU。
每个 epoch 会显示训练、验证和 BLEU 计算进度条。
当前训练入口默认启用：

```text
epochs = 120
batch size = 128
d_model = 256
num_layers = 3
num_heads = 4
d_ff = 512
dropout = 0.1
label smoothing = 0.03
scheduler = warmup_cosine
warmup steps = 500
learning rate = 3e-4
eta_min = 1e-5
validation decode = greedy
early stopping patience = 20
```

模型保存位置：

```text
checkpoints/last.pt
checkpoints/best.pt
```

## 一键实验

运行训练、测试集评估，并自动生成本次实验报告：

```bash
./scripts/run_experiment.sh
```

每次运行会创建一个独立目录：

```text
runs/YYYYmmdd_HHMMSS/
  checkpoints/
  train.log
  eval_testing_greedy.log
  environment.txt
  report.md
```

如果需要覆盖训练参数，直接追加到脚本后面，脚本会原样转发给 `scripts/train.sh`：

```bash
./scripts/run_experiment.sh --epochs 30 --batch-size 128
```

评估阶段默认使用测试集和 greedy 解码。评估参数也可以直接传给一键脚本：

```bash
./scripts/run_experiment.sh --eval-decode beam --eval-beam-size 3
./scripts/run_experiment.sh --eval-split validation --eval-show 5
./scripts/run_experiment.sh --run-id small_warmup_01 --epochs 120 --batch-size 128
```

一键脚本支持的实验/评估参数如下，其余参数都会作为训练参数转发：

```text
--run-id VALUE
--eval-split VALUE
--eval-decode VALUE
--eval-beam-size VALUE
--eval-length-penalty VALUE
--eval-show VALUE
```

这些参数也支持等号写法，例如 `--run-id=small_warmup_01`。

一键脚本生成的 `train.log` 和 `eval_*.log` 会自动过滤 `tqdm` 的动态刷新内容，只保留 epoch 摘要、BLEU 和翻译示例等有效文本。

## 评估

在测试集上计算 BLEU，并输出若干翻译示例：

```bash
./scripts/evaluate.sh
```

也可以通过追加参数覆盖默认设置：

```bash
./scripts/evaluate.sh --checkpoint checkpoints/last.pt --split validation
./scripts/evaluate.sh --decode greedy
```

## 单句翻译

训练完成后，可以使用保存的 checkpoint 翻译单个英文句子：

```bash
./scripts/translate.sh "tom is a student ." --checkpoint checkpoints/best.pt
```

默认使用 beam search，也可以切换为 greedy：

```bash
./scripts/translate.sh "tom is a student ." --decode greedy --checkpoint checkpoints/best.pt
```

## 说明

输入英文需要尽量使用数据集中相同的分词形式，例如标点前留空格：

```text
tom is a student .
```

中文输出会将预测 token 拼接成连续中文字符串。
