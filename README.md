# 基于 Transformer 的英译中机器翻译

> NLP Project, 2026@NJUIS

这是一个课程作业项目：使用 PyTorch 和 `cmn-eng-simple` 数据集，从零训练 Encoder-Decoder Transformer 完成英文到中文翻译。

当前实验的最佳测试集结果为 **BLEU4 29.22**（epoch 54）。完整配置、参数搜索过程和实验分析见 [`docs/experiment_report.md`](docs/experiment_report.md)。

## 项目结构

```text
.
├── data/
│   └── cmn-eng-simple/
│       ├── training.txt
│       ├── validation.txt
│       └── testing.txt
├── docs/
│   ├── experiment_report.md
│   └── question.md
├── scripts/
│   ├── sweep_train.sh
│   ├── sweep_train_full_data.sh
│   ├── sweep_final_from_baseline.sh
│   ├── sweep_params_epoch_window.sh
│   └── train_eval_epoch_window.sh
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

`checkpoints/`、`outputs/` 和由脚本生成的 `data/train_plus_valid.txt` 属于本地训练产物，不纳入 Git 版本控制。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 基础训练命令

```bash
python src/train.py
```

该命令不手动调整任何训练参数，全部使用 `src/train.py` 中的默认配置。默认数据路径为：

```text
train_path=data/cmn-eng-simple/training.txt
valid_path=data/cmn-eng-simple/validation.txt
test_path=data/cmn-eng-simple/testing.txt
```

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
python src/evaluate.py \
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
python src/translate.py \
  --ckpt checkpoints/<run_name>/best.pt \
  --sentence "tom is a student ." \
  --decode beam \
  --beam_size 5 \
  --length_penalty 0.8 \
  --device cuda
```

## 实现要点

- 模型：`torch.nn.Transformer`
- 初始化：随机初始化，从零训练
- 训练：teacher forcing
- 损失：`CrossEntropyLoss(ignore_index=<pad>, label_smoothing=0.03)`
- 优化器：`Adam(lr=3e-4, betas=(0.9, 0.98), eps=1e-9)`
- mask：padding mask + target causal mask
- 解码：greedy 或 beam search
- 指标：BLEU4 + token-level accuracy
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

## 复现最终实验

最终阶段将训练集和验证集合并，并保存 epoch 40–60 的 checkpoint 逐一评估：

```bash
bash scripts/train_eval_epoch_window.sh
```

在 batch size、学习率、dropout 和 label smoothing 的局部范围内搜索时使用：

```bash
bash scripts/sweep_params_epoch_window.sh
```

两个脚本都支持通过环境变量覆盖参数；默认使用 CUDA，并将 checkpoint 和评估结果分别写入 `checkpoints/` 与 `outputs/`。
