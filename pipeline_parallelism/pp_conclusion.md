# Pipeline Parallelism发展历程

# 1、 Gpipe

- [论文链接-EN](https://arxiv.org/pdf/1811.06965)
- [论文链接-CN](https://yiyibooks.cn/arxiv/1811.06965v5/index.html)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在前向传递过程中，GPipe首先将大小为N的每个小批量数据划分为M个相等的微批量数据，这些微批量数据通过K个加速器进行流水线处理。 在反向传递过程中，每个微批次的梯度是基于与前向传递中使用的相同模型参数计算的。 在每个小批量数据结束时，所有M个微批次的梯度都会被累积并应用于更新所有加速器上的模型参数。 此操作序列如图(c)所示。`<br>`

![figure2](images/gpipe-figure2.jpg)

图 (a) 一个示例的神经网络，具有连续的层，被分割到四个加速器上。 $F_{k}$ 是第k个单元的组合前向计算函数。 $B_{k}$ 是反向传播函数，它依赖于来自上一层的 $B_{k+1}$ 和 $F_{k}$ 。(b) 朴素的模型并行策略由于网络的顺序依赖关系导致严重的低利用率。(c) 流水线并行将输入的mini-batch分割成较小的micro-match，使得不同的加速器可以同时处理不同的micro-batch。梯度在最后同步应用。`<br>`

**① 核心思想**：
整个前向做完再去做反向，这样就会形成很大的**Bubble**，
所有的反向做完才做参数的更新，即Update，专业名词**Pipeline-flush**（等所有的micro-batch跑完之后，梯度都累加好之后，再去做统一的梯度更新）。`<br>`
**② 特点**：
当micro-batch个数越多，Bubble所占的比例越低，计算效率越高。`<br>`
**③ 经验**：
micro-batch总的个数至少是stage的四倍以上。`<br>`
**④ 图(c)讲解**：
纵轴是设备device，横轴是时间步step，每个device对应着模型的一个阶段stage。`<br>`

# 2、 PipeDream

- [论文地址-EN](https://arxiv.org/pdf/1806.03377)
- [论文地址-CN](https://yiyibooks.cn/arxiv/1806.03377v1/index.html)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**①** 与传统的单向流水线不同，流水线式的DNN训练涉及到双向流水线。一个小批量数据的前向传递从**输入层开始**，反向传递在**输入层结束**。因此，在流水线中的每个活跃小批量数据可能位于不同的层，无论是在前向传递还是反向传递中。因此，系统中的每台机器必须在以下两个选项之间进行选择：

1. 执行一个小批量数据的前向传递，从而将该小批量数据推送到下游机器；`<br>`
2. 为**另一个**小批量数据执行反向传递，从而确保学习中的前进。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;始终优先考虑前向工作的简单调度机制会阻碍整体的前向进展，因为只有在完成反向传播后才能应用权重更新。同样地，始终优先考虑后向工作可能会导致周期性地存在空闲机器没有可用工作。我们提出了一种避免这些问题的调度机制。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在启动阶段，输入阶段允许NOAM小批量以保持流水线处于稳定状态。一旦进入稳定状态，**每个阶段(stage)会交替执行小批量的前向传播和后向传播。我们称这种机制为一前一后（1F1B）**。在平衡的流水线中，1F1B确保在稳定状态下没有GPU处于空闲(idle)状态，并且我们从每个小批量中取得学习的前进。`<br>`

![figure8](images/pipedream-figure8.png)

**②** 图8展示了一个具有4个阶段的流水线在每个阶段上运行于一台机器的计算时间轴。此配置的NOAM为4。在**启动阶段**，输入阶段接受了恰好四个小批量，并将它们传播到输出阶段。**一旦输出阶段完成了第一个小批量的前向传播，它就会进行相同小批量的后向传播，并开始交替执行后续小批量的前向传播和后向传播**。随着后向传播开始向流水线中的较早阶段传播，每个阶段开始交替执行不同小批量的前向传播和后向传播。如图所示，在稳定状态下，每台机器要么在进行小批量的前向传播，要么在进行小批量的后向传播，保持忙碌。对于1F1B要发挥作用，前向传播的时间不一定要与后向传播的时间相同。事实上，实践中我们观察到后向传播总是比前向传播时间长，而1F1B仍然是一种有效的调度机制。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当阶段(stage)在数据并行配置下运行，并在多个GPU上复制时，我们使用确定性的轮询(round-robin)负载均衡（minibatchID mod stageReplicaID）来将前一阶段的工作分配给复制品。这种确定性的负载均衡确保了对于一个小批量的后向传播是在负责该小批量前向传播的机器上执行的。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;流水线中阶段的1F1B调度策略和跨复制阶段的轮询(round-robin)负载均衡调度策略都是静态策略。因此，它们可以由每台机器独立执行，无需昂贵的分布式协调。`<br>`

**③ 图8核心思想：**
纵向machine1、2、3、4分别对应**stage1、2、3、4**，横向step每个**时间步**，
machine1第一个时间步塞入1号micro-batch，第二个时间步塞入2号micro-batch，第三个时间步塞入3号micro-batch，第四个时间步塞入4号micro-batch，**每个时间步所用到的weight可能不一样**，当machine4做完1号micro-batch的前向后，采用**1f1b**的思想（一个前向一个反向，反向做完就可以释放一部分的activition，是显存节约的一种方式），machine4第二个时间步可以做1号micro-batch的反向也可以做2号micro-batch的前向，它是交叉的来进行的，**有反向就优先做反向，没有就做前向**（这边**实现了Zero Bubble的方式**），后面就没有pipeline-flush，因为**采用了异步更新**的方式，这些micro-batch谁管谁的weight，比如4号batch走到machine3，此时发现1号batch已经做完反向了，此时立刻更新4号它的weight，那么4号就在更新后的weight上继续往下走，在machine3的3号batch用的weight是之前的weight，此时3号和4号用到的weight是**两份不同的weight**，这时候就会**出现weight的冗余**（只是时间不一样，在本论文中**要同时保存，这就是一个致命的缺点**），虽然后期没有bubble但是会多保留weight，**本文工程上不会使用**。`<br>`

# 3、PipeDream-1f1b

- [论文链接-EN](https://arxiv.org/pdf/2006.09503)
- [论文链接-CN](https://yiyibooks.cn/arxiv/2006.09503v3/index.html)
  
![figure8](images/PipeDream-Flush.jpg)

- **①、工程上一定会使用带Pipeline flush的**，没有完全实现异步更新（每个micro-batch自己管自己的更新），异步更新在工程上是不太合理的，
- **②、图（b）中是等4个micro-batch全部做完反向后，统一做更新**，前反向相互交叉进行，最后在同一个weight上面做梯度累加，所有的micro-batch累加完之后，在总的global batch上面去做Pipeline flush参数更新（**这个方式已经在工程上落地了**，在这个落地的情况下，**Megatron-LM在这个基础上又进行了扩展'Interleave-1f1b'**）


# 4、Interleave-1f1b（Megatron-LM）

- [论文地址-EN](https://arxiv.org/pdf/2104.04473)
- [论文地址-CN](https://yiyibooks.cn/arxiv/2104.04473v5/index.html)

    ![figure8](images/Interleave-1f1b-figure.png)

- ①、上半部分的图就是**PipeDream-1f1b中的图（b）**,下半部分就是**Megatron-LM在此基础上改进的Interleave-1f1b，这个方式也在工程上落地了**
- **②、核心思想：**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（1）拆stage的时候用了更细粒度的stage，**一张卡上可以包含两个不同的stage**。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（2）**假设现在有4个Gpu、模型有16层：**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上面图的切分，每个Device放了4层：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（0-3层属于stage0，对应Gpu0，放到Device1上）、
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（4-7层属于stage1，对应Gpu1，放到Device2上）、
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（8-11层属于stage2，对应Gpu2，放到Device3上）、
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（12-15层属于stage3，对应Gpu3，放到Device4上）。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;下面图Interleave的会有一个变化：每个Device放了4层，但是是**间隔的放**，比如：Device1放了0、1、8、9层，中间有明显的间隔，这个间隔就是用不同颜色来表示的，深蓝色的1号micro-batch进入了Device1里，运行第0和1层，运行后立即把output交给Device2，一直交到Device4结束后，此时总共运行了八层，现在该运行第九层了即第8层，第8层在Device1上（浅蓝色的1号micro-batch），采用Interleave的方式，每张卡上是交错的方式，这样做的好处是：上面图Device1是要运行四层才能把output交给Device2的，但是下面图只需要运行两层就能交给下一个Device，那么图中左边灰色的小方格即Bubble就会减半。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（3）节约Bubble时间，但通讯的次数变多

# 5、zero-Bubble

- [论文地址-EN](https://arxiv.org/pdf/2401.10241)
- [论文地址-CN](https://yiyibooks.cn/arxiv/2401.10241v1/index.html)

    ![figure8](images/zero-Bubble-figure.png)

- 图中**F代表Forward，B代表Backward，W代表Wight Backward**，把Backward拆成了两部分B和W。`<br>`
- 用 Linear 来举例，这边主打的是 Linear 的Backward，它需要计算两个部分的梯度：activition、weight（Linear前向时是activition * weight = output,做反向时两个都需要计算梯度）



# 4 图像分类

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;作为概念验证，我们首先使用GPipe对AmoebaNet进行了扩展。我们增加了AmoebaNet中的通道数，并将输入图像大小扩展到480×480。我们使用与[12]中描述的相同超参数，在ImageNet 2012数据集上训练了这个拥有5.57亿参数的AmoebaNet-B(18, 512)。网络被分为4个分区。这个单一模型在单图裁剪的情况下达到了84.4%的top-1和97%的top-5验证准确率 `<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们进一步通过迁移学习[22, 23]在其他图像数据集上展示了巨型卷积网络的有效性。具体而言，我们使用预训练的ImageNet模型，在从通用到细粒度分类的各种目标数据集上进行微调。我们将最后一个softmax分类层的输出单元数更改为目标数据集中的类别数，并随机初始化新的softmax层。所有其他层都使用ImageNet预训练的初始化。在训练期间，网络的输入图像被调整为480×480，随机水平翻转，并使用cutout[24]进行数据增强。训练的超参数与ImageNet相同（我们在补充材料中提供了我们的训练设置的详细描述）。在表4中，我们报告了每个数据集进行了5次微调运行的平均单图裁剪测试准确率。我们的巨型模型在所有目标数据集上获得了有竞争力的结果。例如，CIFAR-10的错误率降低到了1%，CIFAR-100的错误率降低到了8.7%。这些结果证实了Kornblith等人的发现[25]，即更好的ImageNet模型具有更好的迁移性能。`<br>`

![table4](images/gpipe-table4.jpg)

*(表格4：使用在ImageNet 2012上首先训练，然后在其他数据集上进行微调的AmoebaNet-B (18, 512)的图像分类准确率。有关我们训练设置的详细描述，请参阅补充材料。我们的微调结果是在5次微调运行中进行平均的。来自Real等人[12]和Cubuk等人[26]的基准结果是直接从头开始训练的。*Mahajan等人的模型[27]在非公开的Instagram数据上进行了预训练，达到了85.4%的top-1准确率。Ngiam等人[28]通过使用私有数据集（JFT-300M）进行预训练获得了更好的结果。)*

# 5 大规模多语言机器翻译

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;接下来，我们通过扩展自然语言处理（NLP）中使用的模型来展示GPipe的灵活性。由于有大量可用的平行语料库(corpora)，神经机器翻译（NMT）已成为用于NLP的任何架构的基准任务[33, 15, 34, 35, 36]。因此，我们在一个大规模的多语言NMT任务上继续进行GPipe实验。我们使用包含102种语言和英语的平行文档语料库，总共包含250亿个训练示例，每种语言的范围从10^4到10^9 [37]。该数据集通过跨越从数据稀缺（低资源）到数据丰富（高资源）的多种语言，为可扩展性实验提供了一个真实的测试基础。我们首次展示了足够大的NMT模型可以**同时学习100多种语言对之间的映射**，并且在所有语言上的性能都优于双语模型。这进一步凸显了拥有高效灵活的模型并行工具的重要性。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们的比较基于在该语料库中训练的单个Transformer [15]的性能。我们通过两个维度来扩展架构，以展示GPipe的灵活性：（i）沿深度增加模型中的层数，（ii）沿宽度增加前馈层中的隐藏维度和多头注意力层中的注意力头数（以及注意力通道数），类似于Shazeer等人[34]的方法。有关我们的数据集、基准线、训练配置和优化超参数的详细描述，请参阅补充材料。`<br>`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们从一个标准的400M参数的Transformer Big模型T(6, 8192, 16)1开始，如Chen等人[35]所述，词汇表大小为64k。在图3中，我们将其性能与一个13亿参数的深层模型T(24, 8192, 16)、一个13亿参数的宽模型T(12, 16384, 32)、一个30亿参数的模型T(32, 16384, 32)和一个60亿参数的模型T(64, 16384, 32)进行了比较。所有模型都同时在所有语言对上进行训练，使用了多语言BERT2[3]中采用的基于温度的采样方法。T(12, 16384, 32)、T(24, 8192, 32)、T(32, 16384, 32)和T(64, 16384, 32)分别在2、4、8和16个加速器上进行分区。`<br>`
*(注释：1T(L, H, A) --> Transformer model with L encoder layers and L decoder layers, hidden dimension of H and A attention heads. The model dimension is fixed to 1024.)* `<br>`
*(https://github.com/google-research/bert/blob/master/multilingual.md)* `<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从图3中，我们可以观察到将模型容量从4亿参数增加到13亿参数会显著提高所有语言的性能。将模型从13亿参数扩展到60亿参数显示出进一步的改进，特别是对于高资源语言，尽管当将模型从13亿参数扩展到30亿和60亿参数时，**递减收益也可观察到**。接下来，我们将根据这些大规模实验的结果讨论一些我们的经验发现。`<br>`

![figure3](images/gpipe-figure3.jpg)

*(图3：随着多语种模型容量的增加，跨所有语言的翻译质量变化。语言按照从左到右训练数据集大小递减的顺序排列。T(L, H, A)表示具有L个编码器层和L个解码器层，隐藏层维度为H，注意力头数为A的Transformer模型的性能。我们注意到，增加模型容量，从400M参数（T(6, 8192, 16)）到1.3B参数（T(24, 8192, 16)），再到6B参数（T(64, 16384, 32)），在所有语言上都显著提高了质量。当与双语基准线进行比较时，我们还注意到对于低资源语言（图表右侧），质量改进非常大，突显了训练多语种模型所带来的显著迁移收益。)* `<br>`
*(低资源语言（Low-resource languages）是指具有非常有限的相关资源（例如数据、语料库、工具和研究人员）的语言)* `<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;深度与宽度的权衡：我们在我们的多语种设置中研究了深度与宽度之间的权衡，并比较了1.3B宽模型T(12, 16384, 32)和1.3B深模型T(24, 8192, 16)的性能。虽然这两个模型在高资源语言（图3左侧）上的质量非常相似，但**深度更深的模型在低资源语言上取得了巨大的优势**，表明**增加模型深度可能更有利于泛化能力**。此外，与400M模型相比，对于低资源语言（图3右侧），1.3B深模型的质量改进几乎与高资源语言的改进程度相当大，这表明**增加深度可能潜在地增加到低资源任务的迁移程度**。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;深度模型的训练难题：虽然深度增加了神经网络的表示(representational)能力，但也增加了优化问题的复杂性。在我们的大规模(large-scale)实验中，我们遇到了严重的训练难题，这是由于尖锐的激活（正峰度）和数据噪声的组合引起的。我们观察到，在经过几千个步骤的训练后，模型的预测会变得极度尖锐，并对噪声非常敏感，这经常导致非有限(无限)或大梯度，最终破坏了学习进展。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了解决这些问题，我们采用了两种方法：`<br>`

1. 根据Zhang等人的研究[38]，我们将所有Transformer前馈层的参数初始化缩小，缩小模型的层数倍。
2. 当逻辑预测（softmax预激活）的幅度超过一定值时，我们对其进行**剪裁**。这两种方法的组合使我们能够减轻由模型深度缩放引起的训练不稳定性。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;大批量训练：由于其简单性，数据并行是扩展神经网络训练的主要方法[39, 40]。我们通过显著增加用于标准Transformer Big训练的批量大小来测试大批量训练的极限。从每批260K个标记开始，我们将有效批量大小增加到4M，并观察高资源语言对德英（其他语言对也可以观察到类似趋势）的验证损失和BLEU分数。此处使用的优化参数与之前的实验相同。据我们所知，每批4M个标记是迄今为止用于训练NMT模型的文献中使用的最大批量大小[41]。表5显示，随着批量大小的增加，这两个指标都显著改善。我们相信进一步增加批量大小可能会带来更多的改进。`<br>`

![table5](images/gpipe-table5.jpg)

# 6个设计特点和权衡

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;已经提出了几种方法来实现高效的大规模模型并行。然而，每种方法都选择了自己的权衡(trade-off)，使其适用于在特定硬件约束下扩展特定架构。在这里，我们强调了几种模型并行方法涉及的各种设计选择和权衡，并以GPipe为基准，比较了在不同硬件约束和架构变体下的灵活性、可扩展性和效率。模型并行的核心思想是将网络划分为不同的计算单元，然后将它们放置在不同的设备上[42, 43, 44, 45]。从概念上讲，这支持将广泛的模型扩展到巨大的容量。然而，这些方法通常存在硬件利用率低和设备通信瓶颈(communication bottlenecks.)的问题。**单程序多数据（SPMD）** 和**流水线并行性**已被提出作为应对这些挑战的解决方案。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mesh-Tensorflow [34]遵循SPMD范例，将用于数据并行的单指令多数据（SIMD）方法**扩展到其他张量维度**。SPMD允许将每个计算分布在多个设备上，使用户能够将单个矩阵乘法的规模（单个层的模型参数）与加速器的数量成**线性比例扩展**。然而，这也引入了加速器之间高通信开销的问题，原因是需要使用类似AllReduce的操作来合并每个并行化矩阵乘法的输出。该方法仅适用于加速器之间连接速度较高的情景。此外，SPMD限制了可以有效扩展的操作类型，将其使用限制在特定的网络架构和机器学习任务集上。例如，在这种范例下，沿卷积层的通道维度进行划分是不高效的，因为通道实际上是完全连接的，而沿空间维度的划分则需要复杂的技术来处理边界区域。尽管SPMD允许通过使每个操作更小来扩展模型的深度，但它要求将每层分割到更多的加速器上，从而进一步增加了设备间的通信开销。`<br>`
*(注释：SPMD（Single Program Multiple Data）：SPMD是一种并行计算范式，它允许多个处理单元（如处理器、加速器等）同时执行相同的程序，但每个处理单元处理的数据可以不同。每个处理单元可以有自己的控制流和数据，但它们执行的是相同的指令序列。SPMD适用于在不同的数据集上执行相似的计算任务，允许每个处理单元独立地处理自己的数据)* `<br>`
*(注释：SIMD（Single Instruction Multiple Data）：SIMD是一种并行计算范式，它允许一条指令同时作用于多个数据元素。在SIMD中，多个处理单元同时执行相同的指令，但每个处理单元处理的数据不同。每个处理单元执行的是相同的操作，但是操作的输入数据可以是不同的。SIMD适用于在大规模数据集上执行相同的计算操作，允许同时对多个数据元素执行相同的操作。)* `<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其他方法尝试利用基于流水线并行的方法来扩展神经网络[46, 47]。应用于神经网络训练的最新流水线并行方法是**PipeDream [48]** ，其目标是减少参数服务器[49]的通信开销。PipeDream将前向传播的执行流程进行流水线化，并将其**与反向传播交替进行**，以最大程度地提高硬件利用率。然而，这种设计存在由**异步反向更新引入的权重陈旧性**。为避免由权重陈旧性引起的优化问题，PipeDream要求在**每个加速器上维护多个版本化的模型参数副本**，以便准确计算梯度更新，这限制了用户扩展到更大模型的能力。`<br>`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GPipe引入了一种新型的流水线并行性，它**在应用单个同步梯度更新之前对micro-batch的执行进行流水线化**。我们的**批次分割流水线并行算法**结合了重新材料化(re-materialization)技术，可以扩展到大量的micro-batch。这样可以最小化冗余开销，而**无需进行异步梯度更新**。GPipe使用户能够将模型大小与所使用的加速器数量线性扩展。**与SPMD不同**，当扩展模型时，流水线并行仅引入**很少的额外通信开销**。设备间的通信仅在每个micro-batch的分区边界处发生，并且引入的通信开销很小，使得GPipe在无法使用高速设备互连的情况下仍然有用。然而，**目前的GPipe假设单个层适应单个加速器的内存要求(Megatron-LM 每这个要求)**。此外，micro-batch分割需要复杂的策略来支持需要**跨批次进行计算的层**（例如，**BatchNorm在训练过程中使用micro-batch（micro-batch）的统计信息，但在评估时累积mini-batch（mini-batch）的统计信息**）。

# 7 结论

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本工作中，我们介绍了GPipe，一个用于训练大规模神经网络的可扩展模型并行库。我们提出并实现了一种新颖的**批次分割流水线并行算法**，该算法使用**同步梯度更新**，实现了高硬件利用率和训练稳定性下的模型并行。我们利用GPipe来训练大规模的卷积和基于Transformer的模型，并在图像分类和多语言机器翻译任务上展示了强大的实证结果。我们强调了GPipe库的三个关键特点：`<br>`

1. 效率：通过使用新颖的批次分割流水线化算法，GPipe在设备数量增加时几乎实现了线性加速。
2. 灵活性：GPipe支持任何可以表示为层序列的深层网络结构。
3. 可靠性：GPipe利用同步梯度下降，并保证在任意分区数量下的一致训练。
