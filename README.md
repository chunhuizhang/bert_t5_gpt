
> bert_t5_gpt

# transformers

![](./imgs/Transformer-encoder-decoder.jpeg)

- 重点看下，decoder 部分的 Multi-head attention 其实是 Masked 的，见图中最右侧的下三角矩阵
    - 这也是 GPT（decoder-only）的模型架构所采用的方式
  
- post vs. pre LayerNorm

![](./imgs/post_pre_ln.jpeg)
