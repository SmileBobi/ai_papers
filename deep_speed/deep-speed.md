# DeepSpeed：面向所有人的大规模模型训练

[官网链接](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

# 0 概述

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在2020.2月份，我们宣布推出了[DeepSpeed](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)，这是一个开源的深度学习训练优化库，以及库中的一项创新的内存优化技术ZeRO（Zero Redundancy Optimizer）。DeepSpeed通过提升规模、速度、成本和易用性，极大地改进了大型模型训练。DeepSpeed使研究人员能够创建图灵自然语言生成（[Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft)）模型，这是当时参数数量达到170亿个并且具有最先进准确性的最大语言模型。在五月份，我们发布了支持最多2000亿参数的[ZeRO-2](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/) (zero-stage2)技术，相比于现有技术，训练速度提高了10倍。同时，我们还发布了一系列计算、输入/输出和收敛优化技术，为最快的BERT训练提供支持。从那时起，我们一直以快速的速度进行创新，推动着深度学习训练速度和规模的界限。我们不断努力创新，不断突破深度学习训练的速度和规模限制。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;今天，我们很高兴分享我们的新进展，这不仅将深度学习训练推向了极限，而且使更多人能够参与其中，从在大型超级计算机上进行训练的数据科学家，到在低端集群甚至单个GPU上进行训练的人。具体而言，DeepSpeed引入了四项新的系统技术，进一步推动了微软AI产品和平台上的大规模AI(AI at Scale)倡议的创新。这些技术具有极高的计算、内存和通信效率，能够支持具有数十亿到数万亿参数的模型训练。这些技术还允许处理极长的输入序列，并可在硬件系统上进行训练，无论是单个GPU、具有数千个GPU的高端集群，还是具有**非常慢的以太网网络的低端集群**。这样的技术使得更多人可以从中受益，并参与到深度学习训练中。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**使用3D并行性进行万亿参数模型训练**：DeepSpeed实现了三种并行性方法的灵活组合——以ZeRO为基础的数据并行性、流水线并行性和张量切片模型并行性。3D并行性能够根据负载需求的变化灵活适应，以超过一万亿个参数的极大模型为动力，同时实现接近完美的内存扩展性和吞吐量扩展性效率。此外，**改进的通信效率**使用户能够在网络带宽有限的常规集群上以**2-7倍的速度训练具有数十亿参数的模型**。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**使用ZeRO-Offload在单个GPU上进行10倍大规模模型训练**：我们扩展了ZeRO-2，利用CPU和GPU内存来训练大规模模型。使用一台配备单个NVIDIA V100 GPU的机器，我们的用户可以运行最多130亿参数的模型，而不会耗尽内存，这比现有方法的规模要大10倍，同时获得有竞争力的吞吐量。这个功能使得亿级参数模型训练民主化，并为许多深度学习从业者探索更大、更好的模型打开了窗口。[阅读论文:zero-offload](https://www.microsoft.com/en-us/research/publication/zero-offload-democratizing-billion-scale-model-training/)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**1-bit Adam**：通过最多5倍的通信量减少实现高效通信的大规模训练：Adam是一种用于训练许多大规模深度学习模型的有效且（可能是最常用的）优化算法。然而，Adam通常与通信高效的优化算法不兼容。因此，在分布式设备间进行扩展时，通信成本可能成为瓶颈。我们引入了一种新的算法，即具有高效实现的1-bit Adam，它在实现在达到类似Adam的收敛效率的同时，将通信量减少了最多5倍。我们观察到在通信受限场景下，分布式训练速度提高了最多3.5倍，使得能够扩展到不同类型的GPU集群和网络中。[论文链接](https://www.microsoft.com/en-us/research/publication/1-bit-adam-communication-efficient-large-scale-training-with-adams-convergence-speed/)

![figure0](images/deepspeed-figure0.jpg)

# 1 3D并行：实现万亿参数模型的规模化

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;随着现代GPU集群上可用计算能力的迅速增长，训练一个具有惊人能力的万亿参数模型不再是遥不可及的梦想，而是即将实现的现实。DeepSpeed结合了三种强大的技术，使得训练万亿规模模型和扩展到数千个GPU成为可能：数据并行训练、模型并行训练和流水线并行训练。这种共生关系将深度学习训练的规模扩展到远远超出每种策略单独提供的范围。3D并行同时解决了训练万亿参数模型面临的两个基本挑战：**内存效率和计算效率**。因此，DeepSpeed可以在不牺牲速度的情况下，将规模扩展到内存中最庞大的模型。`<br>`

## 1.1 内存效率和计算效率对应巨大模型的挑战

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**内存效率**：训练一个**万亿参数模型**所需的内存远远超出单个GPU设备的可用范围。在混合精度下使用Adam优化器进行训练，仅存储模型状态（参数、梯度和优化器状态）就需要**大约16TB的内存**。以NVIDIA A100 GPU为例，其最先进的GPU只有40GB的内存。为了存储模型状态，需要集合**400个**这样的GPU的内存。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;激活值占用的额外内存随批次大小增加而增加。仅使用单位批次大小训练的万亿参数模型会产生超过1TB的激活内存。通过激活值检查点技术（activation checkpointing），可以通过增加计算量将该内存减少到约20GB，但对于训练来说，内存需求仍然过大。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了使模型能够开始训练而不会耗尽内存，必须在可用的多个GPU设备之间高效地分割模型状态和激活值。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**计算效率**：完整训练一个万亿参数模型需要大约5,000zettaflops（即**5后面跟着24个零**；基于OpenAI的[规模计算法则](https://arxiv.org/abs/2001.08361)）。使用4,000个NVIDIA A100 GPU以50%的计算效率训练这样的模型大约需要100天。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;虽然大型超级计算GPU集群可以拥有超过4,000个GPU，但在这种规模上实现高计算效率是具有挑战性的，因为存在批次大小的限制。计算效率随着计算时间增加而增加，而通信时间则相对减少。这个比例与批次大小成正比。然而，**模型可训练的批次大小有一个上限**，超过这个上限，**收敛效率会急剧下降**。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;世界上最大的模型之一，[GPT-3](https://arxiv.org/abs/2005.14165)，使用约**1,500**的批次大小进行训练。使用**4,000个GPU**，即使是批次大小为4,000，每个GPU只能容纳一个批次，并且会限制可扩展性。`<br>`

## 1.2 数据并行、模型并行和流水线并行间的权衡(trade-off)

### 1.2.1 数据并行

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**数据并行**是深度学习中一种普遍使用的技术，它将每个训练数据批次分割给数据并行的工作节点。在反向传播后，需要传递和聚合梯度，以确保优化器采取一致的步骤。数据并行具有多个独特的优势，包括计算效率和最小的实现工作量。然而，数据并行依赖于将批次大小与数据并行工作节点数量进行扩展，如果无限扩展会影响收敛性。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**内存效率**：数据并行在所有工作节点上**复制模型和优化器**，因此**不具备内存效率**。DeepSpeed开发了[ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)，这是一组优化，可以提高数据并行的内存效率。该工作依赖于ZeRO stage-1，该阶段将优化器状态在数据并行工作节点之间进行分区，以减少冗余。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**计算效率**：随着并行度的增加，每个工作节点执行的计算量是恒定的。数据并行可以在**小规模上实现接近完美的扩展性**。然而，在大规模模型或具有低通信带宽的系统上，数据并行中梯度聚合的**通信成本**与模型大小成比例，限制了计算效率。**梯度累积是一种常见策略**，通过进一步增加批次大小，在本地累积梯度之前，在微批次上执行多次前向传播和反向传播，以分摊通信成本，然后再进行**聚合**并采取优化器步骤。`<br>`

### 1.2.2 模型并行

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**模型并行**是一类广泛使用的技术，它将模型的各个层分割到不同的工作节点上。由于其特性，模型并行的计算和通信是针对特定模型架构的，因此可能需要较大的初始实现工作量。DeepSpeed在这项工作中利用了NVIDIA的[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)，用于大规模模型并行的基于Transformer的语言模型。模型并行可以减少与工作节点数量成比例的内存占用。在三种并行化类型中，**模型并行是最内存高效的，但计算效率最低**。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**内存效率**：模型并行可将内存占用按比例减少与工作节点的数量。重要的是，它是**唯一一种**可以减少单个网络层**激活内存**的方法。DeepSpeed通过在模型并行工作节点之间分割激活内存进一步提高内存效率。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**计算效率**：由于每次前向传播和反向传播中**激活值的额外通信**，模型并行的**计算效率较低(通信引起)**。模型并行需要高通信带宽才能高效运行，并且在通信带宽受限的单个节点Node之外无法良好扩展。此外，每个模型并行工作节点**减少了每个通信阶段之间执行的计算量（计算了一部分）**，影响了计算效率。通常将模型并行与数据并行结合使用，以在内存和计算效率之间进行权衡。`<br>`

### 1.2.3 流水线并行(Pipeline parallelism)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**流水线并行**训练引擎被包含在这个DeepSpeed的版本中！流水线并行将模型的层划分为可以并行处理的阶段。当一个阶段完成微批次(micro-batch)的前向传播时，激活内存会传递到流水线中的下一个阶段。类似地，当下一个阶段完成反向传播时，梯度会通过流水线向后传递。为了确保流水线阶段能够并行计算，必须保持**多个微批次同时进行**。已经开发了多种方法（如[PipeDream](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/)）来权衡内存、计算效率和收敛行为。DeepSpeed的方法**通过梯度累积提取并行性**，以保持与传统的数据并行和模型并行训练相同的收敛行为，使用相同的总批次大小。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**内存效率**：流水线并行可以按流水线阶段的**数量**减少内存占用，使得模型大小能够与工作节点数量线性扩展。然而，流水线并行**不能减少每个层激活的内存占用**。此外，每个工作节点**必须存储**所有**正在进行中的微批次的激活值**。实际上，流水线的第一个阶段的激活内存大约与单个微批次的**总激活内存**相同。一个万亿参数的模型需要大约19GB的内存来存储一个微批次的激活值，几乎占用了新的NVIDIA A100 GPU可用内存的一半。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**计算效率**：流水线并行具有**最低的通信量**，因为它只在阶段边界的层之间传递与激活大小成比例的数据。然而，它不能无限扩展。与模型并行类似，增加流水线的大小会减少每个流水线阶段的计算量，从而降低计算与通信的比率。流水线并行还要求其每个阶段具有完美的负载平衡，以实现良好的效率。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此外，流水线并行在每个训练批次的开始和结束时会产生填充和清空流水线的开销。使用梯度累积步骤(and thus batch size)为流水线阶段数量的4倍或8倍，分别可以实现81%和90%的从一个流水线阶段的扩展效率。`<br>`

## 1.3 通过3D并行实现内存和计算效率

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据并行、模型并行和流水线并行各自在提高内存和计算效率方面扮演着特定的角色。图1说明了我们的3D策略，图中用了 `32张卡32层的Layer`做的3D并行方案。`<br>`

- *下图讲解*：比如一个 `input[16,128]`, DP是 `两个Rank`，那么其 `group size = 2`，就把 `input`按照batch size拆成 `两部分[8,128]`, 分别放入两个DP里， **混合并行的时候，一个完整的DP是需要一个完整的模型**（图中模型是32层），图下方说明了有 32 个 workers，此时每个 DP 分别有16个 worker（即Gpu），这16个Gpu是有一套完整的 weight。用Rank 0来解释，**Stage 0有4个Gpu(一种颜色代表一个worker)、8层 Layers，纵轴按照PP来拆分，横轴按照TP来拆分（每一层的weight分到四个不同的Gpu上）**
  `<br>`
- **TP之间是怎么做通信的呢？**
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1、`Rank 0 里的 Stage 0 内部中`：纵轴第一层（四个不同颜色的小方块）做 All Reduce。
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2、`跨DP的通信（不同DP的All Reduce）`：就是（*不同Rank、相同Stage、相同层、颜色相同的小方块*）分别对应做All Reduce（Rank 0 里的 Stage 0 中的第一层的第一个紫色小方块与Rank 1 里的 Stage 0 中的第一层的第一个紫色小方块做All Reduce，绿色的与绿色的做All Reduce，以此类推橙色、蓝色也是这通信，一共做8层的通信）。
- **不同的PP之间做 send 和 recive**，卡和卡是相对应的，做完TP之后，每一部分和每一部分相对应的做传输（Rank 0 中 Stage 0 的0-7层 和 Stage 1 的8-15层的紫色小方块是相对应的做传输，即紫色双向箭头）。`<br>`
- **计算出来的Optimizer是如何分配的？**
- **还用了Zero策略，Zero策略在Optimizer是如何分配的？**
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;就是拆一个小方块，每个Dp里的对应的小方块的Optimizer State是一样的，在此基础上小方块还可以应用Zero策略：就是（*不同Rank、相同Stage、相同层、颜色相同的小方块*）分别对应做Zero（Rank 0 里的 Stage 0 中的第一层的第一个紫色小方块与Rank 1 里的 Stage 0 中的第一层的第一个紫色小方块做Zero，*怎么做Zero呢（它们之间weight是一样的，是保存Optimizer里的Optimizer State，只要保存这个一半的weight就行）*，`Zero是在不同DP之间做的`，图中TP做个4份，所以有4个Zero

![figure1](images/deepspeed-figure1.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**内存效率**：将模型的层划分为流水线阶段，然后通过模型并行将每个阶段的层进一步划分。这种二维组合同时减少了模型、优化器和激活值消耗的内存。然而，我们不能无限地划分模型，否则会产生通信开销，从而限制了计算效率。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**计算效率**：为了使工作节点数量在模型并行和流水线并行之外**继续扩展**，同时又不损失计算效率，我们使用基于ZeRO的数据并行（ZeRO-DP）。ZeRO-DP不仅通过**优化器状态分割**进一步提高内存效率，还通过利用(exploiting)**拓扑感知映射**(topology aware mapping)，使得在最小的通信开销下可以扩展到任意数量的GPU。`<br>`

- `按照图1的方案` Gpu 的分配如下：
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Pipeline Paralle`：就是 Gpu4、12、20、28 这个维度
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Zero Data Parallel`：其属于 `Data Parallel`，**Zero只能在 `Data Parallel`情况下做切分**。
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上面16张卡组成一个DP, 下面16张卡组成另一个DP，Gpu0、8、16、24代表4个不同的Stage，每个Stage有8层，总共32层。
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Model Paralle`：在deepspeed中其实就是TP，绿色的四个小方块分别代表Gpu0、1、2、3，一定是连续的，要保证TP在同一个节点之内。
- 一般是8张卡（Gpu0、1、2、3，4，5，6，7）放在同一个节点，为啥这边Gpu4，5，6，7放到另一个DP里呢，DP之间做All Reduce，All Reduce通信量是比较大的，所以把0、1、2、3，4，5，6，7放入同一节点之内，优先排列TP,再排列DP，最后PP（PP一般是跨节点的，比如Gpu4、12）

![figure2](images/deepspeed-figure2.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;拓扑感知的3D映射（图2）：在3D并行中，每个维度都被精心映射到工作节点上，通过利用两个关键的架构特性，实现最大的计算效率。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**优化节点内和节点间通信带宽**：模型并行具有三种策略中**最大**的通信开销，因此我们优先将模型并行组放置在节点内(同一个node)，以利用更大的节点内部带宽。在这里，我们采用了基于张量切片(tensor-slicing)的模型并行的NVIDIA Megatron-LM。当模型并行未跨越节点中的所有工作节点时，数据并行组放置在节点内。否则，它们将跨节点放置。流水线并行具有**最低**的通信量，因此我们可以在节点之间调度流水线阶段，而不受通信带宽的限制。`<br>`
*(gpus index 排列: 模型并行--> 数据并行 --> 流水线并行； 并行层次：数据并行(outer)--> 流水线并行(middle)--> 模型并行(inner))* `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**通过并行通信实现带宽放大**：通过流水线并行和模型并行，每个**数据并行组**传递的梯度大小线性减小，因此**总通信量从纯数据并行中减少**。此外，每个数据并行组在一组局部化的工作节点中独立并行地执行通信。因此，通过减少通信量、增加局部性和并行性，数据并行通信的有效带宽得到放大。`<br>`

## 1.4 3D 并行是如何使用各并行手段来训练万亿级参数的

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过使用8路模型并行、64路流水线并行和8路数据并行，万亿参数模型可以在**4,096**个NVIDIA A100 GPU上进行扩展。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过结合模型并行和流水线并行，3D并行能够在多个节点上实现出色的内存效率和计算效率。**模型并行为节点内的激活值和模型状态提供了内存效率**，而**流水线并行允许在节点之间实现模型状态的内存效率**，而**不损失计算效率**（相比仅使用模型并行）。在我们的万亿参数示例中，使用微批次(micro batch)大小为1，并结合上述的3D并行策略，我们的模型在模型状态方面将消耗30 GB的内存，在分区的激活值方面将消耗**2.5 GB**的内存（在激活检查点之后）。这将导致总共占用32.5 GB的内存空间。使用这样的配置，具有40 GB内存的NVIDIA A100 GPU足以容纳和训练这样的模型。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将模型并行与流水线并行相结合还可以使流水线并行在非常小的批次大小下实现高计算效率，即使在非常小的批次大小下，通过使用8路模型并行，每个模型的微批次为1，每个GPU的有效微批次为1/8。因此，流水线并行可以通过使用**8倍(1/(1/8) = 8)的流水线并行度**进行梯度累积步骤，并且每个GPU只有1的批次大小，实现90%的计算效率。当与数据并行结合使用时，在4,096个GPU上实现了有效的批次大小为4,096，仍然可以实现90%的流水线效率。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;但是数据并行的计算效率如何呢？难道数据并行需要每个GPU上的大批次(large batch per gpu)才能保持高效吗？`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**模型并行可以将每个GPU的有效批次大小降低到小于1**。这使得流水线并行即使在小批次(mini-batch)大小下也能隐藏流水线泡沫开销。需要注意的是，通过在节点(nodes)之间使用流水线(pipline)并行，在流水线的每个阶段，我们实际上允许数据并行节点之间的通信独立并行地进行，与其他流水线阶段同时进行。事实上，在高端GPU集群中常见的全连接网络拓扑中，这对于数据并行训练的有效通信带宽有着重要的影响。由于流水线的每个阶段的每个节点(node)都可以与其对应的**数据并行节点并行通信**，因此有效的通信带宽与流水线阶段的数量成正比。使用64个流水线并行阶段，有效带宽是与单个节点之间的双向带宽相比的64倍。有了如此大的有效带宽，**流水线并行可以使数据并行有效扩展**，即使在计算与通信比率非常低的小批次大小下也可以实现高效的训练。`<br>`

## 1.5 3D 并行总结

* **3D并行** = **数据并行（DP）** + **流水线并行（PP）** +  **张量并行（TP）** 。
* **目标** ：突破单设备内存限制、降低通信开销、最大化计算资源利用率。
* **典型框架** ：NVIDIA的  **Megatron-DeepSpeed** 、Google的  **Pathways** 。

### **1.5.1 数据并行（Data Parallelism, DP）**

* **核心思想** ：将训练数据划分为多个子批次（mini-batch），分配到不同计算设备（如GPU）上并行处理，最后同步梯度。
* **适用场景** ：模型参数可以完整加载到单个设备内存时。
* **优势** ：
  * 简单易实现，扩展性强。
  * 适合单卡内存足够但需要加速训练的情况。
* **局限性** ：无法解决模型参数过大无法放入单卡内存的问题。
* **示例** ：PyTorch的 `DistributedDataParallel`（DDP）。

### **1.5.2 流水线并行（Pipeline Parallelism, PP）**

* **核心思想** ：将模型按层切分为多个阶段（stage），每个阶段部署到不同的设备上，按顺序执行（类似工厂流水线），同时通过微批次（micro-batch）技术减少设备空闲时间。
* **适用场景** ：模型层数较多，单设备无法容纳完整模型。
* **优势** ：
  * 减少单设备内存占用。
  * 通过流水线调度提升设备利用率。
* **局限性** ：需要设计合理的切分策略，避免流水线气泡（bubble）过大。
* **示例** ：GPipe、PipeDream。

### **1.5.3 张量并行（Tensor Parallelism, TP）**

* **核心思想** ：将单个张量运算（如矩阵乘法）拆分到多个设备上执行，例如按行或列切分权重矩阵，分布式计算后合并结果。
* **适用场景** ：模型层内参数极大（如大矩阵运算）。
* **优势** ：
  * 进一步降低单设备内存需求。
  * 支持超大规模参数的高效计算。
* **局限性** ：通信开销较大，需优化设备间数据传输。
* **示例** ：NVIDIA的 **Megatron-LM**（用于Transformer层的切分）。

### **1.5.4 3D并行的协同工作**

* **结合方式** ：
  * 将数据并行（DP）用于切分数据批次。
  * 流水线并行（PP）用于切分模型层。
  * 张量并行（TP）用于切分单个层的计算。
* **实际应用** ：
  * 在训练 **GPT-3** 时，同时使用：
    * **数据并行** 切分数据到多个计算节点。
    * **流水线并行** 切分模型层到多个设备组。
    * **张量并行** 切分每层的矩阵运算到组内设备。
  * 通过三者结合，实现超大规模模型的分布式训练。

## 1.6 3D并行通过线性效率扩展为万亿参数模型训练提供动力

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; DeepSpeed可以使用**仅800个NVIDIA V100 GPU训练**具有**一万亿参数**的语言模型（图3）。我们通过扩展模型的大小观察到线性增长，无论是模型的大小还是训练的吞吐量，从而展示了同时的内存和计算效率。在每个配置中，我们**可以在每个GPU(32G)上训练大约14亿个参数**，这是单个GPU在不耗尽内存的情况下所支持的最大模型大小，表明内存扩展达到了完美的线性比例。我们还获得接近完美线性的计算效率扩展和**每个V100 GPU的吞吐量为47 TeraFLOPS**。这对于给定的硬件来说是令人印象深刻的扩展和吞吐量。`<br>`

![figure3](images/deepspeed-figure3.jpg)

### 1.6.1 深入分析3D 并行是如何加速GPT-3级别的训练的

![figure4](images/deepspeed-figure4.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在图4中，我们使用了最近的[GPT-3](https://arxiv.org/abs/2005.14165)模型架构作为1750亿参数的基准，用于评估3D并行性：`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先，我们评估了2D配置（C1-C3）。配置C1和C2**仅使用流水线和模型并行性**，它们可以训练模型，但由于过度分解问题和低GPU利用率，吞吐量较低。配置C3尝试仅使用流水线和数据并行性，但在**不通过Megatron的模型并行性减小激活值大小**的情况下，无法将问题适应内存中。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3D配置（C4-C10）按照流水线并行性的增加程度排列；在平衡并行性以实现内存、计算和通信效率方面，**中间配置**实现了最佳性能。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最佳的3D方法**每个GPU可实现49 TeraFLOPS的吞吐量**，超过理论硬件峰值的40%。`<br>`

### 1.6.2 混合并行如何在低带宽集群上将GPT-2的训练加速高达7倍

![figure5](images/deepspeed-figure5.jpg)

*(注释：低节点带宽：GPU之间用带宽50Gbps的节点内带宽连接，节点之间使用4Gbps的节点间带宽连接。)*  `<br>`
*(注释：A100的NVLink-v3带宽：600Gbps 单向; v100 d Nvlink-v2: 300Gbps 单向; )* `<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们在图5中展示了混合并行在训练一个拥有**15亿参数**的GPT-2模型时的通信优势。我们在一个具有低节点间带宽的集群的四个节点上进行训练，以突出训练中的通信阶段：`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在这种情况下，由于低节点内带宽和**较小的模型大小，模型并行并不具有优势**。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**流水线并行的通信量比数据并行和模型并行配置少一个数量级**，并且在小批次大小下的速度是7倍。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据并行使用**梯度累积**来分摊随着批次大小增加而产生的通信开销，但是在更大的批次大小下，流水线并行配置的性能仍然是数据并行的**两倍**以上。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;混合的流水线并行和数据并行配置通过**将数据并行组限制在节点内的GPU上**，避免了梯度通信瓶颈，因此梯度通信可以受益于更快的节点内带宽。`<br>`

# 2. ZeRO-Offload：使用单个GPU进行10倍更大模型的训练

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Offload通过利用GPU和宿主CPU上的计算和内存资源，推动了可以通过**最小的GPU资源**高效训练的最大模型规模的边界。它允许在**单个NVIDIA V100 GPU上训练高达130亿参数的模型**，比现有技术大10倍，同时保持每个GPU**超过30 TeraFLOPS**的高训练吞吐量。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过在单个GPU上实现数十亿参数模型的训练，ZeRO-Offload使大型模型训练民主化，使其对资源有限的深度学习实践者可获得。`<br>`

![figure6](images/deepspeed-figure6.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Offload的关键技术是我们在[ZeRO-2](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/)的基础上实现的新能力，即将**优化器状态和梯度**卸载到CPU内存中。这种方法使得ZeRO-Offload能够**最小化**CPU卸载所带来的计算效率损失，同时实现与原始ZeRO-2相同甚至更好的效率。下图显示了ZeRO-Offload的架构。`<br>`

![figure7](images/deepspeed-figure7.jpg)

## 2.1 ZeRO-Offload 如何在单GPU卡上实现数十亿参数模型的训练的

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;训练像GPT和T5这样的**百亿**参数模型需要许多GPU来适应模型及其状态在GPU内存中的存储。大型模型训练通常使用多个GPU设备上的**模型并行性**来解决内存限制问题。最近，我们发布了ZeRO，这是一种内存高效的优化器，它将模型状态（优化器状态、梯度和模型权重）在数据并行的GPU之间进行分割，使得百亿参数模型可以在**不需要模型并行性**的情况下进行训练。然而，ZeRO仍然需要大量的数据并行GPU来**存储分割的模型状态**，限制了只有少数人可以访问这些资源的大型模型训练。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Offload通过使其即使在**单个GPU上**也可以进行大型模型训练，使大型模型训练民主化。为了在不使用多个GPU的情况下训练百亿参数模型，ZeRO-Offload继承了ZeRO-2的优化器状态和梯度分割。与ZeRO-2不同，ZeRO-Offload将**优化器状态和梯度都卸载到主机CPU内存中**。优化器状态在整个训练过程中都保留在CPU内存中。梯度则在反向传播过程中使用GPU上的reduce-scatter进行计算和平均，并且每个数据并行进程将属于其分割的平均梯度卸载到CPU内存中（在图7中的g offload），同时丢弃其余部分。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一旦梯度在CPU上可用，每个数据并行进程直接在CPU上并行**更新优化器状态分割**（在图7中的p update）。更新完成后，参数分割移回GPU，并在GPU上进行all-gather操作以收集所有更新的参数（在图7中的g swap）。ZeRO-Offload还**利用CUDA流**在通信（如g offload和g swap）和计算（如反向传播和p update）之间**实现重叠**，以最大化训练效率。`<br>`

## 2.2 ZeRO-Offload 在模型规模、训练速度和可扩展性方面的收益

![figure8](images/deepspeed-figure8.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10倍的模型规模：在单个32 GB的V100 GPU上，图6显示PyTorch可以训练的最大模型具有**13亿**个参数，而ZeRO-Offload允许训练具有130亿参数的模型，规模比之前大10倍。这是因为ZeRO-Offload在整个训练过程中将优化器状态（占用大部分GPU内存）保存在主机内存中，同时将梯度在反向传播过程中卸载到CPU。因此，保存下来的GPU内存可以用于承载更大的模型进行训练。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;高效的训练吞吐量：图8显示，在训练一个100亿参数的模型时，即使只使用单个GPU，ZeRO-Offload每个GPU提供超过**30 TeraFLOPS**的吞吐量，并且随着GPU数量的增加，其吞吐量几乎线性增加。`<br>`
*(v100 fp16 峰值算力：112 TFLOPS; A100 fp16 峰值算力：312 TFLOPS)*  `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Offload很好地补充了ZeRO-2，在少量GPU上支持大型模型的高效训练。从1到16个GPU，ZeRO-Offload通过利用CPU内存将模型训练从不可行变为可行，减少了模型所需的GPU内存。**在32个GPU上，ZeRO-Offload略优于ZeRO-2**；这是因为ZeRO-Offload在GPU上节省了额外的内存，使得可以使用**更大的批次大小**进行训练，并**提高了GPU的计算效率**，尽管有CPU卸载的开销。随着GPU数量的增加（如64和128个），ZeRO-2优于ZeRO-Offload，因为两者现在可以运行相似的批次大小。一方面，ZeRO-2没有将数据移动到CPU的开销，另一方面，GPU上的优化器步骤计算比在CPU上快得多。总之，ZeRO-Offload很好地补充了ZeRO-2，并扩展了ZeRO优化系列，覆盖了从单个设备到数千个设备的大型模型训练的完整范围。`<br>`

# 3. DeepSpeed稀疏注意力：以6倍更快的执行速度支持比原先长10倍的序列

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;基于注意力机制的深度学习模型，例如Transformer，在捕捉输入序列中的token之间的关系方面非常有效，即使跨越较长的距离。因此，它们被应用于文本、图像和基于声音的输入，其中序列长度可以达到数千个token。然而，尽管注意力模块在捕捉长期依赖关系方面非常有效，但在实践中，它们在处理长序列输入时受到计算和内存需求的限制，这些需求随着序列长度的增加呈**二次增长**。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了解决这个限制，DeepSpeed提供了一套稀疏注意力内核(sparse attention kernels)，通过块稀疏(block-sparse)计算的方式，可以将注意力计算的计算和内存需求降低数个数量级。该套件不仅减轻了注意力计算的内存瓶颈，还能高效地执行稀疏计算。其API允许方便地与任何基于Transformer的模型集成。除了提供广泛的稀疏结构，它还具有处理任何用户定义的块稀疏结构的灵活性。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;具体来说，稀疏注意力（SA）可以设计为在附近的tokens之间计算局部注意力，或者通过使用局部注意力计算的摘要token进行全局注意力。此外，SA还可以允许随机注意力或局部、全局和随机注意力的任意组合，如图10中的蓝色、橙色和绿色块所示。因此，SA将内存占用减少到 $o(wn)$ ，其中 $1 < w < n$ 是一个参数，其值取决于注意力结构。`<br>`

![figure10](images/deepspeed-figure10.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**在GPU上的高效实现**：尽管稀疏注意力的基本实现可能节省了内存，但在计算方面，它甚至可能比完全计算更差。这主要是由于稀疏数据在整体计算中引入了分歧(divergence)和非协调的内存访问。总的来说，在GPU上开发高效的稀疏内核是具有挑战性的。DeepSpeed提供了在[Triton](https://github.com/ptillet/triton)中开发的高效稀疏注意力内核。这些内核采用了块稀疏的模式，可以实现对齐的内存访问，减轻线程分歧，并平衡处理器上的工作负载。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**系统性能**：如图11所示，稀疏注意力（SA）在长序列上具有超过10倍的能力，并且计算速度最高可提高6.3倍。左图展示了在BERT-Base和BERT-Large模型中可运行的最长序列长度，分为三个设置：密集（dense）、带有激活检查点的密集（dense with activation checkpoint）和稀疏（SA）带有激活检查点。与密集相比，SA使BERT-Base和BERT-Large的序列长度分别增加了10倍和16倍。此外，与密集相比，SA减少了总计算量并提高了训练速度：随着序列长度的增加，提速效果更显著，对于BERT-Base可以提高高达6.3倍，对于BERT-Large可以提高5.3倍。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**系统性能**：如图11所示，稀疏注意力（SA）在超过10倍长度的序列上具有强大的能力，并且计算速度最高可提高6.3倍。左图显示了在BERT-Base和BERT-Large模型下三种设置下可运行的最长序列长度：密集（dense）、带有激活检查点的密集（dense with activation checkpoint）和稀疏（SA）带有激活检查点。与密集相比，SA分别使BERT-Base和BERT-Large的序列长度增加了10倍和16倍。此外，与密集相比，SA减少了总计算量并提高了训练速度：随着序列长度的增加，提速效果更为显著，对于BERT-Base可以提高高达6.3倍，对于BERT-Large可以提高5.3倍。`<br>`

![figure11](images/deepspeed-figure11.jpg)

## 3.1 稀疏注意力(SA) 是如何达到接近或超过 full-attention的精度的

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;相关的研究工作，如稀疏注意力([Sparse Transformer](https://arxiv.org/pdf/1904.10509.pdf))、[Longformer](https://arxiv.org/pdf/2004.05150.pdf)和[BigBird](https://arxiv.org/pdf/2007.14062.pdf)，在稀疏注意力方面已经展示出与完全注意力相媲美甚至更高的准确性(accuracy)。我们的经验也与这些研究相一致。除了更低的内存开销和更快的计算速度外，我们还观察到在实际生产模型中，稀疏注意力可以实现**更高的准确性和更快的收敛速度**。以下图表展示了基于BERT进行长文档理解（2,048个序列长度）的生产模型训练的准确性。实验在三种设置下进行：从头开始的密集训练、从头开始的稀疏训练以及从使用512个序列长度的密集模型检查点继续训练的稀疏训练。我们观察到，在从头开始的预训练中，稀疏训练相比密集训练更快地收敛，并且准确性更高。此外，从使用稀疏模型进行预训练的检查点继续训练的结果表现得更好，无论是在时间还是准确性方面。 `<br>`

![figure12](images/deepspeed-figure12.jpg)

## 3.2 稀疏注意力(SA) 与sota的 Longtransformer 比较

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们将SA与Longformer进行了比较，Longformer是一种最先进的稀疏结构和实现方法。在我们的实验中，SA使用了"[Fixed](https://arxiv.org/pdf/1904.10509.pdf)"稀疏性，并且这两种实现方法在准确性方面是可比较的。在系统性能方面，SA在预训练和推理中都优于Longformer：`<br>`

- 在Wikitext103上进行MLM预训练的执行速度比Longformer快1.5倍。
- 在BERT-Base上进行推理（批量大小为1，序列长度为2,048）的执行速度比Longformer快3倍。

## 3.3 能够处理任何块稀疏结构的灵活性

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DeepSpeed稀疏注意力套件并不针对任何特定的稀疏结构，而是为模型科学家提供了有效的系统支持，使其能够探索任何块稀疏结构。目前，我们已经添加了一些流行的稀疏结构，例如[Fixed](https://arxiv.org/pdf/1904.10509.pdf)（来自OpenAI Sparse Transformer）、[BigBird](https://arxiv.org/pdf/2007.14062.pdf)（来自Google）和BSLongformer（AI2 [Longformer](https://arxiv.org/pdf/2004.05150.pdf)的块稀疏实现）。我们还定义了一个“可变”结构的模板，如图10所示，可以用于简单地自定义任何块稀疏的随机、局部或全局注意力模式。`<br>`

# 4. [1 bit Adam](https://www.microsoft.com/en-us/research/publication/1-bit-adam-communication-efficient-large-scale-training-with-adams-convergence-speed/)：通信减少5倍，训练速度提高3.4倍

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;大规模模型（如BERT和GPT-3）的可扩展训练需要在模型设计、架构和系统能力方面进行精心优化。从系统角度来看，**通信已成为一个主要瓶颈**，特别是在具有标准TCP互连的普通系统上，这些系统的网络带宽有限。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**通信压缩**是减少这类系统上训练时间的重要技术。其中一种最有效的压缩通信方式是通过**误差补偿压缩**，它提供了鲁棒的收敛速度，即使在进行1位压缩时也能保持良好的效果。然而，最先进的误差补偿技术只适用于基本优化器，如随机梯度下降（SGD）和带动量的SGD，这些优化器在梯度上是线性相关的。它们无法与像Adam这样的非线性梯度优化器一起使用，而Adam在许多任务中，包括BERT类模型的训练中，提供了最先进的收敛效率和准确性。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于像Adam这样强大的优化器来说，梯度的非线性依赖性（在方差项中）使得开发基于误差补偿的压缩技术变得具有挑战性，从而限制了最先进的通信压缩技术的实际价值。`<br>`

## 4.1 经典压缩技术背景介绍

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通信压缩的一种方式是1 bit 压缩，可以表示为：`<br>`

$$
x\rightarrow \frac{|x|}{|Sign(x)|} Sign(x)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过这种压缩方法，我们可以通过使用1位来表示每个数字，从而实现32倍的内存大小减小。然而，使用这种简单的方法会显著降低收敛速度，导致该方法不适用。为了解决这个问题，最近的研究表明，通过使用误差补偿压缩，我们可以期望在通信压缩的基础上几乎获得相同的收敛速度。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;误差补偿的思想可以总结为：1）进行压缩，2）记录压缩误差，然后3）在下一次迭代中将压缩误差添加回去。对于随机梯度下降（SGD），进行误差压缩的结果是:`<br>`

$$
x_{t}=x_{t-1}-\gamma C(g_{t}+e_{t-1}), \quad e_{t}=g_{t}+e_{t-1}-C(g_{t}+e_{t-1})
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中，C(.)表示1bit压缩操作符。通过进行误差补偿，好处在于历史的压缩误差 $e_{t}$ 和 $e_{t-1}$ 最终会相互抵消，这可以通过以下方式看出：`<br>`

$$
x_{t}=x_{t-1}-\gamma(g_{t}+e_{t-1}-e_{t})
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这种策略已经被证明适用于所有线性依赖于梯度的优化算法，例如随机梯度下降（SGD）和动量SGD。`<br>`

## 4.2 Adam 优化器误差补偿的挑战

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们在下面提供Adam算法的概述。更新规则如下：`<br>`

$$
m_{t+1}=\beta_{1} m_{t}+(1-\beta_{1}) g_{t} \dots (momentum term)$$ 
$$v_{t+1}=\beta_{2} v_{t}+(1-\beta_{2})(g_{t})^{2} \dots (variance term)
$$

$$
x_{t+1}=x_{t}-\gamma \frac{m_{t+1}}{\sqrt{v_{t+1}}+\eta}
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如上述公式所示，方差项 $v_{t}$ 与梯度 $g_{t}$ 非线性相关。如果我们将基本的误差补偿压缩应用于Adam算法，我们观察到Adam将无法收敛，如图13所示。`<br>`

![figure13](images/deepspeed-figure13.jpg)

## 4.3 使用1 bit Adam进行通信压缩

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了在使用Adam优化器时进行通信压缩，我们开发了1 Bit Adam，通过预处理来解决梯度中的非线性问题。我们观察到，在经过几个训练轮次后，非线性项方差 v_{t} 的变化幅度显著减小，并且在之后将其保持不变不会改变收敛速度。所提出的1bit Adam优化器如图14所示，由两个部分组成：热身阶段（warmup stage）和压缩阶段（compression stage）。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;热身阶段本质上是原始的Adam算法，而压缩阶段则保持**方差项不变**，并将**剩余的线性项**（即动量）压缩为1位表示。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;算法的压缩阶段受到阈值参数的控制（如图14所示）。当我们检测到“方差”**变化低于一定阈值时**，我们切换到压缩阶段。我们的研究表明，整个训练过程中只需要进行15-20%的热身阶段即可。`<br>`

![figure14](images/deepspeed-figure14.jpg)

### 4.3.1 Adam 压缩的深入分析

对于1 bit Adam中的权重更新规则由以下公式控制。对于第 $i_{th}$ 个工作节点，在压缩阶段中：`<br>`

![formula1](images/deepspeed-formula1.jpg)

## 4.4 解决1 Bit Adam在系统中的挑战

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;除了算法上的挑战之外，在训练系统中应用1位Adam还存在两个系统挑战。首先，我们需要高效的内核将**动量转换为1位表示**。其次，我们需要高效的通信方案在不同的GPU之间交换这个压缩的动量。压缩的目标是减少整体的训练时间，以便可以使用带宽受限的互连系统来训练大型模型。我们在DeepSpeed中解决了这些挑战，并引入了一个针对受通信限制的系统进行训练的完全优化的1 Bit Adam实现。`<br>`

## 4.5 1Bit Adam在受通信限制的系统上的优势

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 Bit Adam提供与Adam相同的收敛性，并且通信开销最多**减少5倍**，可以实现BERT-Large预训练的吞吐量提高最多**3.5倍**，以及SQuAD微调的吞吐量提高最多2.7倍。这种端到端的吞吐量改善是通过在压缩阶段观察到的6.6倍（图15左）和6.2倍（图15右）的加速实现的。值得一提的是，我们的1 bit Adam优化器在40 Gigabit以太网系统上的扩展性非常好，其性能与在40 Gigabit InfiniBand QDR系统上的Adam相当。我们注意到，基于iPerf基准测试，40 Gigabit以太网的有效带宽为4.1 Gbps，而基于InfiniBand perftest微基准测试，InfiniBand提供接近峰值带宽32 Gbps的性能。`<br>`

![figure15](images/deepspeed-figure15.jpg)

## 4.6 1 Bit Adam 性能案例研究

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与Adam相同的收敛性：使用1Bit Adam的一个主要问题是收敛速度，我们发现1bit Adam可以在相同数量的训练样本下达到与Adam相同的收敛速度和可比较的测试性能，如图16所示。

![figure16](images/deepspeed-figure16.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BERT-Base和BERT-Large的详细结果如表1所示。我们可以看到，在未压缩和压缩的情况下，得分与原始模型相当或更好。`<br>`

![table1](images/deepspeed-table1.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最多减少5倍的通信量：1位Adam提供与Adam相同的收敛性，并且在16位（FP16）训练的压缩阶段将通信量减少了16倍。对于BERT预训练，这导致整体通信量减少了5倍，因为我们观察到热身阶段只占总体训练时间的15%。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算原始Adam与1位Adam之间通信量比例的公式如下：`<br>`

$$
1 / (warmup + (1 - warmup)/16)
$$

*(warmup 指的时warmp 的 adam 所占的比例)*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1Bit Adam在BERT-Large训练中速度提升3.5倍：我们在两种带宽受限的互连系统上展示了在训练BERT-Large时的两个主要结果：1) 40 Gbps以太网（图17左）和2) 40 Gbps InfiniBand QDR（图17右）。在压缩阶段，我们观察到以太网系统的吞吐量最多提高了6.6倍，而InfiniBand系统的吞吐量最多提高了2倍，从而分别实现了3.5倍和2.7倍的端到端加速（包括热身阶段和压缩阶段）。1 Bit Adam的主要优点来自通信量的减少，这得益于我们的压缩动量交换以及我们自定义的allreduce操作，它使用非阻塞gather操作后跟allgather操作来实现高效的1 bit通信。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;需要注意的是，通过使用像LAMB这样的优化器并增加总批量大小，可以减少通信。然而，根据我们的经验，对于大批量的情况，严格的超参数调整往往更加困难。此外，1 bit Adam对于具有小关键批量大小（无法使用大批量获得良好收敛性）的工作负载也非常有效，例如许多微调任务。`<br>`

![figure17](images/deepspeed-figure17.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1位Adam在SQuAD微调中速度提升2.7倍：1位Adam不仅在大规模训练任务上具有可扩展性，而且在像SQuAD微调这样的任务上也表现出色。如图18所示，1bit Adam在基于以太网和InfiniBand的系统上都具有良好的扩展性，并在基于以太网的系统上达到最高6.2倍的吞吐量（在压缩阶段），从而实现了2.7倍的端到端加速（25%热身阶段加上75%压缩阶段）。对于SQuAD微调，我们观察到总批量大小为96时具有最佳的F1分数。大于此值的批量大小会降低收敛速度，并需要额外的超参数调整。因此，为了扩展到32个GPU，我们每个GPU只能使用3-4个较小的批量大小。这使得微调任务具有较高的通信需求，并且难以扩展。1bit Adam很好地解决了扩展挑战，实现了3.4倍的通信量减少而无需增大批量大小，并实现了2.7倍的端到端加速。`<br>`

![figure18](images/deepspeed-figure18.jpg)

请访问[DeepSpeed网站](https://www.microsoft.com/en-us/research/project/deepspeed/)和[Github 仓库](https://github.com/microsoft/DeepSpeed)以获取有关这些新技术的代码、教程和文档！我们还将其中一些技术集成到ONNX Runtime中（在新标签页中打开）。`<br>`
