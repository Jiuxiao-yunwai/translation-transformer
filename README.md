# 基于 Transformer 的英译中机器翻译

这是一个课程作业项目：使用 PyTorch 和 `cmn-eng-simple` 数据集，从零训练 Encoder-Decoder Transformer 完成英文到中文翻译。

项目不使用预训练模型、预训练词向量、HuggingFace 或外部翻译模型。BLEU4 在 `src/metrics/bleu.py` 中手写实现，不调用现成 BLEU 函数。

## 项目结构

```text
.
├── data/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── cmn-eng-simple/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── translate.py
│   ├── data/
│   │   └── dataset.py
│   ├── vocab/
│   │   └── vocab.py
│   ├── models/
│   │   ├── positional_encoding.py
│   │   └── seq2seq_transformer.py
│   ├── decoding/
│   │   ├── greedy.py
│   │   └── beam_search.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── metrics/
│   │   └── bleu.py
│   └── utils/
│       └── runtime.py
├── checkpoints/
│   └── <run_name>/
│       └── best.pt
├── outputs/
│   └── <run_name>/
│       ├── test_predictions.txt
│       └── result.json
└── requirements.txt
```

`src/` 已按职责拆分：一个核心类一个文件，例如 `Vocab`、`PositionalEncoding`、`Seq2SeqTransformer` 分别独立放置。

## 数据格式

每行格式必须是：

```text
english sentence<TAB>chinese sentence
```

英文处理：

```python
english.lower().strip().split()
```

中文处理：

```python
chinese.strip().split()
```

中文不使用 jieba，也不重新按字切分。默认过滤长度超过 `80` 的源句或目标句。

特殊符号固定为：

```text
<pad>=0, <unk>=1, <bos>=2, <eos>=3
```

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 基础训练命令

```bash
cd /home/jiuxiao-u/nlp/translation-transformer

/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python src/train.py \
  --train_path data/cmn-eng-simple/training.txt \
  --valid_path data/cmn-eng-simple/validation.txt \
  --test_path data/cmn-eng-simple/testing.txt \
  --epochs 120 \
  --batch_size 128 \
  --num_workers 2 \
  --d_model 256 \
  --nhead 4 \
  --num_layers 3 \
  --dim_feedforward 512 \
  --dropout 0.1 \
  --lr 3e-4 \
  --label_smoothing 0.03 \
  --decode beam \
  --beam_size 5 \
  --length_penalty 0.8 \
  --device cuda
```

## 冲 BLEU4 29+ 的建议命令

```bash
/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python src/train.py \
  --train_path data/cmn-eng-simple/training.txt \
  --valid_path data/cmn-eng-simple/validation.txt \
  --test_path data/cmn-eng-simple/testing.txt \
  --epochs 160 \
  --batch_size 96 \
  --num_workers 2 \
  --d_model 384 \
  --nhead 6 \
  --num_layers 3 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --lr 3e-4 \
  --scheduler warmup_cosine \
  --warmup_steps 800 \
  --eta_min 1e-5 \
  --label_smoothing 0.03 \
  --decode beam \
  --beam_size 5 \
  --length_penalty 0.8 \
  --device cuda
```

显存不足时，把 `--batch_size 96` 改成 `64`。

训练时每个 epoch 会输出：

```text
train_loss, valid_loss, valid_bleu, valid_token_acc, lr
```

训练、验证和测试评估都会显示 tqdm 进度条。需要关闭时加：

```bash
--no-progress
```

每次训练都会自动创建一个实验名，例如：

```text
20260609_183012_dm384_bs96_beam
```

对应输出位置：

```text
checkpoints/<run_name>/best.pt
outputs/<run_name>/result.json
outputs/<run_name>/test_predictions.txt
```

训练结束后查看本次实验结果：

```bash
cat outputs/<run_name>/result.json
head -60 outputs/<run_name>/test_predictions.txt
```

最终测试阶段会打印并写入：

```text
test_bleu1, test_bleu2, test_bleu3, test_bleu4, test_token_accuracy
```

也可以手动指定实验名：

```bash
--run_name dm384_beam5_lp08_run01
```

## 单独评估

```bash
/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python src/evaluate.py \
  --ckpt checkpoints/<run_name>/best.pt \
  --test_path data/cmn-eng-simple/testing.txt \
  --run_name eval_<run_name> \
  --decode beam \
  --beam_size 5 \
  --length_penalty 0.8 \
  --device cuda
```

## 单句翻译

```bash
/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python src/translate.py \
  --ckpt checkpoints/<run_name>/best.pt \
  --sentence "tom is a student ." \
  --decode beam \
  --beam_size 5 \
  --length_penalty 0.8 \
  --device cuda
```

## 参数搜索脚本

训练参数 sweep：

```bash
bash scripts/sweep_train.sh
```

它会自动生成带时间戳的实验名，例如：

```text
20260609_223000_dm384_bs128_lr3e4_do01
```

结果分别写到：

```text
checkpoints/<run_name>/best.pt
outputs/<run_name>/result.json
outputs/<run_name>/train.log
```

可以用环境变量临时覆盖部分设置：

```bash
SWEEP_EPOCHS=120 DEVICE=cuda NUM_WORKERS=2 bash scripts/sweep_train.sh
```

对某个 checkpoint 扫 beam 解码参数：

```bash
bash scripts/sweep_decode.sh checkpoints/<run_name>/best.pt
```

它按两步执行：

```text
1. 固定 beam_size=5，搜索 length_penalty = 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0
2. 自动读取第 1 步最好的 length_penalty，再搜索 beam_size = 3, 4, 5, 8
```

使用训练集和验证集合并训练的参数 sweep：

```bash
bash scripts/sweep_train_full_data.sh
```

训练 sweep 默认最多跑 `50` 个 epoch，并启用早停。需要更久时可以覆盖：

```bash
SWEEP_EPOCHS=120 bash scripts/sweep_train_full_data.sh
```

这个脚本会自动生成：

```text
data/train_plus_valid.txt
```

内容是：

```text
data/cmn-eng-simple/training.txt + data/cmn-eng-simple/validation.txt
```

注意：因为 validation 也被放进训练集，`best_valid_bleu` 会偏乐观。这个脚本主要看最终 `test_bleu4`。

该脚本默认用 `last.pt` 做最终测试：

```text
--final_ckpt last
```

这样不会依赖已经参与训练的 validation 来选择 `best.pt`。

它会自动尝试多组参数，包括：

```text
nhead = 6, 8
num_layers = 3, 4
batch_size = 48, 64, 96, 128
d_model = 384, 512
dim_feedforward = 1024, 1536
lr = 3e-4, 2e-4
dropout = 0.1, 0.15
```

基于当前最佳配置的最终训练局部搜索：

```bash
bash scripts/sweep_final_from_baseline.sh
```

该脚本以 `dm512_h8_l4_bs48_ff1536_lr2e4_do015` 为基准，固定：

```text
train = training + validation
final_ckpt = last
test_decode = beam
bleu_every = 999
early_stopping = off
```

默认运行第二阶段，只搜索 dropout 和 label smoothing：

```text
A: dropout=0.12, label_smoothing=0.03
B: dropout=0.18, label_smoothing=0.03
C: dropout=0.15, label_smoothing=0.02
D: dropout=0.15, label_smoothing=0.04
E: dropout=0.15, label_smoothing=0.03
```

第三阶段学习率搜索：

```bash
STAGE=lr DROPOUT=0.15 LABEL_SMOOTHING=0.03 bash scripts/sweep_final_from_baseline.sh
```

会搜索：

```text
lr = 1.5e-4, 2e-4, 2.5e-4, 3e-4
```

第四阶段比较 3 层和 4 层：

```bash
STAGE=layers DROPOUT=0.15 LABEL_SMOOTHING=0.03 bash scripts/sweep_final_from_baseline.sh
```

固定 epoch 对比：

```bash
STAGE=epochs bash scripts/sweep_final_from_baseline.sh
```

会搜索：

```text
epoch = 43, 47, 50, 55
```

脚本会按 batch size 自动设置 warmup：

```text
batch_size=48  -> warmup_steps=400
batch_size=64  -> warmup_steps=400
batch_size=96  -> warmup_steps=300
batch_size=128 -> warmup_steps=300
```

## 实现要点

- 模型：`torch.nn.Transformer`
- 初始化：随机初始化，从零训练
- 训练：teacher forcing
- 损失：`CrossEntropyLoss(ignore_index=<pad>, label_smoothing=0.03)`
- 优化器：`Adam(lr=3e-4, betas=(0.9, 0.98), eps=1e-9)`
- mask：padding mask + target causal mask
- 解码：greedy 或 beam search
- 指标：手写 BLEU4 + token-level accuracy
- 输出：`checkpoints/best.pt`、`outputs/test_predictions.txt`、`outputs/result.json`

## 调参建议

如果 loss 不下降：

- 检查 causal mask 和 padding mask。
- 学习率从 `3e-4` 改成 `1e-4`。
- 检查 decoder input 和 decoder target 是否错位。

如果 train loss 下降但 BLEU 低：

- 检查中文是否被错误切字。
- 检查 BLEU 是否过滤 `<bos>`、`<eos>`、`<pad>`。
- 检查 greedy/beam 是否正常遇到 `<eos>` 停止。

如果 GPU 功率跑不满：

- 把 `--num_workers 0` 改成 `--num_workers 2`。
- 训练阶段可以先用 `--decode greedy` 加快验证，并加 `--test_decode beam` 让最终测试仍然使用 beam。
- 显存足够时增大 `--batch_size`。

早停默认关闭。需要启用时加：

```bash
--early_stopping_patience 20 --min_delta 0.01
```

含义是 `valid_bleu` 连续 20 个 epoch 没有提升超过 0.01 就停止训练。

默认每个 epoch 都计算验证 BLEU。需要每隔 N 个 epoch 才计算一次时加：

```bash
--bleu_every 5
```

训练 sweep 脚本也支持环境变量：

```bash
BLEU_EVERY=5 bash scripts/sweep_train_full_data.sh
```
