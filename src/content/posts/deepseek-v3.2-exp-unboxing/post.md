---
title: DeepSeek-V3.2-Exp 源码开箱
published: 2025-12-28
description: DeepSeek-V3.2-Exp 开箱研究体验：单 GPU 运行最小可研究模型，以及一些杂谈。
# image: ./cover.webp
tags: [LLM, DeepSeek]
category: 笔记
draft: false
slug: deepseek-v3.2-exp-unboxing
---

# LLM 初体验

月初在 NV Nemo-AutoModel 套件上动手实作了 Qwen 模型的微调训练，尝试读了一下代码，感觉 LLM 的模型代码确实非常简单清晰，AutoModel 框架也非常 pythonic。
总结代码阅读的体验就是：用成熟的 PyTorch 和 Transformer 库拼装模型本体的三个组件：token embedding + positional encoding、Attnetion block、FFN 或 MoE，对计算复杂的部分，例如 FlashAttn 或者 low-rank MoE，patch 一个优化的算子，这个算子可能是由 native CUDA、Triton、TileLang 等语言或工具进行实现的，然后通过一份 config recipe，由通用的训练、微调、推理脚本处理，对接 huggingface 的模型、数据集等资源开始工作。

最近这几天，想着从主流大模型的模型架构设计上找一些能用的模块和点子，做到神经着色里面来，最近趁着脑子比较浆糊，就看了一些 tile-based GPU 编程的相关内容，接触到了 Tilelang、cuTile（CUDA 13）、CuteDSL 这个几个语言，打算往下精进下去。那么开源至宝 DeepSeek 自然是非常优秀的学习材料。


# DeepSeek-V3.2-Exp 开箱

让我们直接开始！首先把环境准备好，然后初步研读代码，并设法建立一个单卡友好的研究原型版本。

## 环境准备

来到 [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) 的仓库，简单浏览了一下目录结构，非常整洁清爽，没有多余的 submodule 负担。

快速翻阅 README：DeepSeek-V3.2-Exp 基于 V3.1-Terminus 打造，进一步优化了 token 的效率，直觉是应该又改进了稀疏化的效率，细读文字材料发现确实如此：推出了 DeepSeek Sparse Attention，旨在以更低的成本获得不逊色于 V3.1-Termius 的性能。

翻看 inference 目录，看到非常熟悉的几个源文件，这份源码此前在 TileLang 仓库中以例子的形式呈现：[tilelang/examples/deepseek_v32/inference](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32/inference)。说明未来 DeepSeek infra 的算子应该都会基于 TileLang 来编写，进一步精进 tiled-based HPC 是 AI infra 的必备技能。

查看 inference/requirements.txt，依旧是非常清爽的依赖库，没有指定具体 torch 版本，需要注意的是引入了一个额外优化的算子 [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)，进一步移步检查，该算子主要实现三种精度下的 
`Multiply each row of x by the Hadamard transform matrix.`
查看了该库的 wheel release，发现没有 torch>2.2、cu>12.2、python>=3.12 的分发包，因此无法避免 CUDA 编译。

使用之前创建的 TileLang 环境做基底：torch2.9.1cu12.8、tilelang 0.1.7、ninja 1.13.0、nvcc v12.8.93，
尝试直接 `pip install -r requirements.txt` 或 `pip install fast-hadamard-transform` 是会收到报错的，大概的意图好像是 fast-hadamard-transform 会现在 release wheels 里面找到预编译的轮子，找不到就自动下载源码进行编译，编译过程大概率会报错一个 `import torch` 但是找不到 torch 的错误，原因好像是不会使用（不会检查？）当前的 venv 进行插件编译。

我的解决方式是直接下载 fast-hadamard-transform 源码，`pip install --no-build-isolation .` 使用这个进行安装（不知道什么时候开始的，后续建议所有插件编译都带上 `--no-build-isolation` 参数），这样在我的 Debian 12 和 服务器的 22.04 LTS，可以直接安装 built cache。

## 初步研读

自己研究神经网络模型的时候，跑通 forward pass 对我理解整个模型非常重要，单步调试 + debug console 可以非常清晰的了解整个模型的运作。
由于没有多卡的分布式设备，我要设法在单张 GPU 上完成这个过程，同时也不需要真的推理 671B 级别的 DeepSeek 模型，只需要把模型框架搞明白即可，因此不需要准备权重，要设法传入参数构造最小可运行模型。

所有的 Transformer 模块全都在一个 [model.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py) 文件中，构造模型的参数类 [ModelArgs](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L17-L90) 也均含有推荐的默认值，没有随意置空超参数，还标注好了分类，
加速算子全部在 [kernel.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py) 中使用 TileLang 实现，
完整模型参数也全部在 [config_671B_v3.2.json](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json) 可读，
甚至没有多余的 utils 代码，
非常清爽、整洁、干净。
更棒的是，而且在 main 函数入口就提供了可以即刻启动的 forward pass 的推理测试流程： [inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L916-L923)

```python
if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
```

DeepSeek 这份代码是一个模型实现很棒的例子，不管给初学者、老手、高手，都有使用代码的自信和欲望，甚至可以在没有运行环境下都可以阅读和学习这份代码。

顺便骂一嘴。多年经验积累，我觉得某些自诩灵活的、模块化的网络模型工具箱，都有面向对象、设计模式的软件工程大病。
这些模型库喜欢把并不复杂的模型拆散成各种模块类，然后放到不同的源文件，模块一多自然就很细碎。
然后用所谓的工厂模式，把所有的模块参数全部聚合到配置文件里，有时还妄图在配置文件中去组织连通性，用不明所以的脚本去产生模型的各个模块。
静态阅读和调试起来非常灾难，尤其对我这种笨蛋和新手而言非常不友好。

它们都忽视了一个最重要的准则，神经网络本质就是清晰的张量数据流、计算图。
请时刻谨记：一个模型永远对应一个完整的输入到输出链路，它就应该是少量/单个源码上完整可读的、可调试的，而不应该使用配置文件去保证模型之间的连通性，配置文件分类组织好参数、管理好实验就够了。
多余的软工病，都是蠢笨的行为。

简言之，网络模型本来就是一进一出的事，那为何不把单个模型全部放到单个文件，形成**一个模型对应一个模型文件、一个模型文件一个配置文件**的超清晰模式呢？
人脑这点空间本来就不擅长处理长记忆长连接，为什么非要组织结构上故弄玄虚呢？嘶……感觉好像想到某些图形渲染的库也有同样的毛病，也该骂。

继续阅读代码。

直接使用 `python model.py` 跑起来，果然 24GB 的显存直接炸掉了（好像需要使用 32.66GB 的显存），意料之中的错误。

进入 ModelArgs 类中找到参数说明 `n_layers (int): Number of transformer layers.`，默认值为27层，将层数降低至2层，保证重复又足够小。
此外我还将最大的单词表大小，从 102400 缩小到 2048，只是为了提升速度，后经实测不修改也可以。

```python ins={2-3}
args = ModelArgs()
args.n_layers = 2
args.vocab_size = 2048
```

继续运行起来，根据 ModelArgs 构造完 Transformer 模型本体后，直接进入模型类的 [inference forward](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L889-L913)：

```python {"1. Token embedding":11} {"2. Transformer layers loop":17} {"3. Distributed linear layer":22} {"4. Distributed gathering":25}
@torch.inference_mode()
def forward(self, tokens: torch.Tensor, start_pos: int = 0):
    """
    Forward pass for the Transformer model.
     Args:
        tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
        start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.
     Returns:
        torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
    """

    seqlen = tokens.size(1)
    freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
    mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1) if seqlen > 1 else None
    h, residual = self.embed(tokens), None


    for layer in self.layers:
        h, residual = layer(h, residual, start_pos, freqs_cis, mask)
    h, _ = self.norm(h, residual)


    logits = self.head(h[:, -1].float())


    if world_size > 1:
        all_logits = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(all_logits, logits)
        logits = torch.cat(all_logits, dim=-1)
    
    return logits
```

继续执行，抛出了在执行到 MLA (Multi-head Latent Attention) 模块里面的 Indexer 模块时出现运行时错误 
`RuntimeError: kernel fp8_index_kernel_ input q_s ndim expected 3, but got 4`, 这是第一个指向 TileLang fp8_index 算子的错误。

简单翻阅了 MLA 模块的 forward，看起来像是做完 KV cache 之后有一个对 cache 的评分，然后筛选出评分最高的部分进行后续计算。
后续询问 AI 得知，这是在实现 low-rank 压缩之后，动态筛选高相关度 tokens 的做法，以达到更稀疏、高效的相关度标记，这就是传说中 DeepSeek Sparse Attention 的关键算子。
此外，注释里也说明 `# we use fp8 kv cache in actual deployment, so here we simulate the precision by casting kv to fp8 and then back to bf16.` 实际采用 FP8 低精度部署的 KV cache，对应后续的 fp8_indexer TileLang kernel，或者说这是一个 index score applier。

算子调用位于：[`index_score = fp8_index(...)`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L480)

算子错误陷于：[`fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L274)

在 kernel.py 源码中，看到了熟悉的 TileLang JIT kernel 的双括号单行启动：`fp8_index_kernel(q.shape[2], q.shape[3])` 通过传入 tile shape 相关的参数构造 kernel，`(q, q_s, k, k_s)` 则是启动 kernel 传入的计算数据。通过字母猜测，q 和 q_s 可能是评分相关的张量，然后应用各种奇妙的矩阵乘和 top-k max 计算在 key 上的评分。

因为 tile-based 的 GPU 编程模型中 shape 是 layout 最关键的构造参数之一，TileLang 也是如此，后续参与计算的 tensor shape 肯定是会定义清楚的。
所以为了修复常见的 shape 不对的 runtime error，需要继续阅读 kernel 的实现。
从 `fp8_index(...)` 调用处继续跟踪，就看到了真正的实现，也就是装饰有 `@T.prim_func` 的函数。

```python {4, 6}
@T.prim_func
def fp8_index_kernel_(
    q: T.Tensor[(b, m, h, d), FP8],
    q_s: T.Tensor[(b, m, h), FP32],
    k: T.Tensor[(b, n, d), FP8],
    k_s: T.Tensor[(b, n), FP32],
    o: T.Tensor[(b, m, n), FP32],
) -> None:
```

这样就清晰了，甚至根据这个 shape 都能大概猜到 GEMM 会怎么做，返回到 fp8_kernel 的调用，通过 debug console 检查所有 q、k 的形状。
q_s 和 k_s shape 第四维是 1，需要将其去掉才能对上 kernel 的 tensor shape 定义。

```python {2, 4}
q.shape = torch.Size([2, 128, 64, 128])
q_s.shape = torch.Size([2, 128, 64, 1])
k.shape = torch.Size([2, 128, 128])
k_s.shape = torch.Size([2, 128, 1])
```
为了保证算子的源码的整洁，我不打算在 kernel.py 的调用处进行，而是将 model.py Indexer 模块部分做修改。

```python del={1} ins={2-5}
index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(), self.k_scale_cache[:bsz, :end_pos].contiguous())
index_score = fp8_index(q=q_fp8.contiguous(),
                        q_s=weights.squeeze(-1).contiguous(),
                        k=self.k_cache[:bsz, :end_pos].contiguous(),
                        k_s=self.k_scale_cache[:bsz, :end_pos].squeeze(-1).contiguous())
```

重新运行，Indexer 模块又炸了 `ValueError: Default process group has not been initialized, please make sure to call init_process_group.`，这次是要将 topk indices broadcast 到分布式组里面。

```python
import torch.distributed as dist
# ...
dist.broadcast(topk_indices_, src=0)
```

嗯，可以想到，这是一个在所有的多 GPU 切片上共享相同的评分索引（top-k indicies），保证后续模型在多 GPU 上计算的一致性，
这也是 Transformer 模块的 inference forward 最后一步要从各个分布式节点上 gather 最终结果的原因。

于是，用 DeepSeek 写了一个 init 的片段，运行之后警告没有 destroy，所以我自己补了一个，用这两个片段包住主函数内的代码。
其中 world_size、rank、block_size 的设置是源码默认提供的，不做修改，我曾尝试使用 torchrun 也跑不起来（好像我用 torchrun 就没跑起来过任何东西）。

```python
# DeepSeek-V3.2-Exp pre-settings
world_size = 1
rank = 0
block_size = 128

# Torch distributed init
if not dist.is_initialized():
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='tcp://127.0.0.1:23456',
        world_size=world_size,
        rank=rank
    )

# Torch distributed destroy
if dist.is_initialized():
    dist.destroy_process_group()
```

终于跑通了，结果应该非常清爽，没有任何错误和警告提示，对应输入和输出的 tensor shape，如下：

```
torch.Size([2, 128])
torch.Size([2, 2048])
```

至此，就完成了 DeepSeek-V3.2-Exp 模型的修改，可以完美的进行 forward pass。若要继续研究和实验，这会是一个非常棒的开始。

::github{repo="Gemuxi/DeepSeek-V3.2-Exp-SM89"}


## 其他问题

其实还是花了一下午和一个晚上的时间，来跑通和初步理解这份代码的，期间确实遇到了不少问题，走了一点弯路，也记录一下。

当 fast-hadamard-transform 源码编译安装失败的时候，我起初丝毫没有想到是 `--no-build-isolation` 参数的问题。
因为最近遇到的包，基本还是可以直接使用 pip 安装的，或者会告诉我需要添加，不知道什么时候需要、什么时候不需要，后续可能我都会加上了。

看到作者给出了等价实现，并且还是一个一行过的一把梭，`F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale`，就想着 hook 一个函数，于是就想着用 code agent 阅读 fast-hadamard-transform 代码做上下文后代劳修改，结果是惨败。code agent 把这个 hadamard 函数写得非常繁琐，然而实际上只需要两三行即可。
在我替换了正确的版本后，fp8_indexer kernel 出现的 tensor shape 问题，我也想用 code agent 进行自动修复。结果是也花了很长的时间，也没能找到正确的地方进行修改，甚至几度 agent 还想要介入修改 kernel.py 的源码，尝试修改算子，显然这是不合理的行为，而且很显然现在的 code agent 没有 TileLang 这种 DSL 的语言知识，所写的代码肯定完全不可用。所以事实是，处理这些错误，人的直觉和经验确实更高效，或许直接向 AI 问答都比全自动 code agent 更靠谱。

此外，我观察到，DeepSeek-V3.2-Exp 这份代码实际上在 RTX3090 GPU 运行会产生不可修复的错误，同样来自 fp8_indexer 算子：

```sh wrap showLineNumbers=false
/home/mu/miniforge3/envs/tile/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include/cute/arch/mma_sm89.hpp:88: static void cute::SM89_16x8x32_F32E4M3E4M3F32_TN::fma(float &, float &, float &, float &, const unsigned int &, const unsigned int &, const unsigned int &, const unsigned int &, const unsigned int &, const unsigned int &, const float &, const float &, const float &, const float &): block: [0,43,0], thread: [96,0,0] Assertion `0 && "Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED"` failed.
```

其中 SM89 就是 RTX4090 的 compute capability，所以更换了 RTX4090 GPU 才能把代码运行起来。
这个问题应该可以通过修改算子 kernel 的配置进行适配解决，考虑部署的角度，其他的通用推理框架部署也没有问题，而从研究角度出发，我暂时止步于此。
这个报错使我意识到，通用的编程模型和硬件支持固然重要，而 DeepSeek 走的路线就是要把硬件和模型绑定在一起，当作一个整体的系统进行迭代和调优。
我非常青睐这种技术风格。
在未来，市场上没准真的会出现像游戏卡带一样，硬件上烧录了模型的即插即用的专用硬件呢？感觉会很有意思。

这是我第一次接触到 tiled-based DSL 的实际例子，我信赖编译器的自动优化，以及对调试体验的巨大提升，以及超级清晰的代码。
在我看来，在新的 cuTile 编程模型对低于 Blackwell 架构以下的硬件全面开放之前，TileLang 和 CuteDSL 是非常有学习价值的。
上周看的 [CuteDSL 讲座](https://www.bilibili.com/video/BV1cjyEBhEPu/) 也提到可以尝试着把 CuteDSL 当作普通的 CUDA 来写，甚至 NV 都给 TileLang 开发了 CuteDSL backend，可见 tiled-based GPU 编程模型在未来会有超大的潜力。

