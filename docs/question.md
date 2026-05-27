# 课程作业：基于Transformer的机器翻译系统

## 一、问题描述

利用Transformer，将输入的英文句翻译成中文。

## 二、数据集

- **输入**：一句英文（e.g. `tom is a student .`）
- **输出**：中文翻译（e.g. `汤姆是个学生。`）
- **训练集**：18000句
- **验证集**：500句
- **测试集**：2636句


## 三、要求

- TensorFlow 或者 PyTorch 实现

---

## 四、评价指标

### BLEU score

&gt; BiLingual Evaluation Understudy, IBM

将机器翻译产生的候选译文与人翻译的多个参考译文相比较，越接近，候选译文的正确率越高。

统计同时出现在系统译文和参考译文中的 n 元词的个数，最后把匹配到的 n 元词的数目除以系统译文的 n 元词数目，得到评测结果。

### BLEU 计算公式

$$
BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

其中：

- **$N$**：最大语法的阶数，实际取 4。
- **$w_n = 1/N$**：权重。
- **$p_n$**：出现在答案译文中的 n 元词语接续组占候选译文中 n 元词语接续组总数的比例。
- **$BP$**：长度过短句子的惩罚因子（Brevity Penalty）。

$$
BP = 
\begin{cases} 
1 & \text{if } c &gt; r \\ 
e^{(1-r/c)} & \text{if } c \leq r 
\end{cases}
$$

- **$c$**：候选译文中单词的个数。
- **$r$**：答案译文中与 $c$ 最接近的译文单词个数。

&gt; **BLEU 分值范围：0 ~ 1**，分值越高表示译文质量越好，分值越小，译文质量越差。

---

## 五、参考资料

1. 《神经网络与深度学习》第 15.4.2 小节
2. [Attention Is All You Need (Harvard NLP 注释版)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and Polosukhin, I.. **Attention is all you need**. In NIPS 2017.
4. Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D. F., & Chao, L. S. (2019, July). **Learning Deep Transformer Models for Machine Translation**. In ACL 2019.

---

&gt; 《自然语言处理课件》by 虞剑飞