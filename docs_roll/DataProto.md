# DataProto

`DataProto` 是 ROLL 中最核心的批数据协议，定义在 `roll/distributed/scheduler/protocol.py` 中，被用作数据集整理、rollout、奖励计算、分布式调度以及 actor/critic 训练之间的统一载体。

从工程实践上看，`DataProto` 不是一个简单 DTO，而是贯穿大部分训练系统的中间表示。

## 为什么需要 DataProto

ROLL 需要一个对象，能够同时承载以下几类信息：

* 按 batch 维对齐、可直接送入模型的 tensor 字段
* 每个样本一份、但不适合表示为 tensor 的非 tensor 字段，例如字符串、dict、UUID、多模态对象
* 批级别的运行控制信息，例如 `micro_batch_size`、`generation_config`、`metrics`
* 切分、拼接、分组、重排、分布式回收等操作能力

单纯的 `dict[str, torch.Tensor]` 无法满足这些需求，因此 `DataProto` 被设计为模块之间的统一协议。

## 结构

`DataProto` 由三层组成：

```python
DataProto(
    batch: TensorDict,
    non_tensor_batch: Dict[str, np.ndarray],
    meta_info: Dict,
)
```

### `batch`

`batch` 用来存放按样本对齐的 tensor 字段，通常是模型输入或训练目标，例如：

* `input_ids`
* `attention_mask`
* `position_ids`
* `responses`
* `old_log_probs`
* `ref_log_probs`
* `advantages`
* `values`

这里使用 `TensorDict` 的原因是，整批数据可以天然支持切片、堆叠、分块和 dataloader 迭代。

### `non_tensor_batch`

`non_tensor_batch` 用于存放每个样本一份、但不适合表示为 tensor 的数据。其值必须是 `np.ndarray(dtype=object)`，并且长度必须和 batch size 一致。

典型示例包括：

* `sample_uuid`
* `answer`
* `domain`
* `multi_modal_inputs`
* `multi_modal_data`

这在多模态和 agentic 场景里尤其重要，因为有些信息只会在 strategy 边界被进一步转换成 tensor。

### `meta_info`

`meta_info` 存放批级上下文，而不是样本级负载。

典型字段包括：

* `global_step`
* `micro_batch_size`
* `generation_config`
* `forward_args`
* `metrics`
* 动态批处理相关元数据，例如 `global_micro_batch_indices`

可以把它理解成当前 batch 的协议头。

## 约束条件

`DataProto` 强制维护几条重要约束：

* 只支持一个 batch 维
* 当 `non_tensor_batch` 非空时，每个字段长度都必须等于 batch size
* `non_tensor_batch` 的值必须是 `np.ndarray(dtype=object)`
* `from_dict()` 中所有 tensor 的前导 batch shape 必须一致

这些约束决定了 `DataProto` 本质上是“样本表”，而不是一个通用嵌套容器。

## 构造方式

最常见的构造入口有两个：

* `DataProto.from_single_dict()`
* `DataProto.from_dict()`

它们会自动把字段拆分到 tensor 和 non-tensor 两部分，并完成一致性校验。

典型用法如下：

```python
batch = DataProto.from_single_dict(
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer": np.array(answers, dtype=object),
    },
    meta_info={"global_step": 10},
)
```

## 核心语义

理解 `DataProto` 时，有两个合并语义最重要。

### `concat`：增加样本

`DataProto.concat()` 会沿 batch 维拼接多个 batch。

适用场景包括：

* 合并不同 domain 的 rollout 结果
* 合并不同 rank 的 Ray worker 输出
* 合并 dynamic batching 生成的 shard
* 为了对齐 batch size 而追加重复样本

它是按“行”扩展数据。

### `union`：增加字段

`DataProto.union()` 用于合并两个 batch size 相同的 `DataProto`，把缺失字段补到另一侧。

适用场景是样本集合不变，但字段集合变多。

它是按“列”扩展数据。

可以简单记成：

* `concat` 表示更多行
* `union` 表示更多列

## 常用操作

`DataProto` 提供了一组非常关键的批处理 API：

* `slice()` 和 `__getitem__()`：按范围取样本
* `select_idxs()`：按索引或布尔 mask 选样本
* `chunk()`：把一个 batch 分成多个 `DataProto`
* `reorder()`：原地重排样本顺序
* `group_by()`：按 tensor 或 non-tensor 字段分组
* `repeat()`：重复样本
* `pop()`：从当前对象中移除字段并返回
* `rename()`：重命名 batch 中的 key
* `clone()`：深拷贝
* `to(device)`：迁移 tensor 负载
* `make_iterator()`：通过 `DataLoader` 迭代 mini-batch

正是这些 API 让 `DataProto` 成为一个可操作的 pipeline IR，而不只是一个被动容器。

## `concat` 中的元信息聚合

`concat()` 对 `meta_info` 的处理方式与 batch 负载不同：

* 非全局 key 默认保留 rank0 的值
* 全局 key 会跨输入聚合
* 默认情况下，`metrics` 会被当作全局 key

如果某个全局 key 本身是 dict，则会按子 key 聚合。这非常适合分布式指标收集场景：每个 rank 返回局部指标，pipeline 最终只需要一个合并后的 `DataProto`。

## 序列化与 Ray

`DataProto` 实现了 `__getstate__` 和 `__setstate__`，因此可以比较自然地在 Ray worker 之间传输。

`DataProto.materialize_concat()` 是标准辅助函数，用来：

* 从 Ray `ObjectRef` 中拉回 `DataProto`
* 在需要时过滤包装后的引用
* 把所有结果合并成一个 batch

这意味着在 ROLL 中，`DataProto` 基本可以看作很多分布式训练阶段的内部 RPC 消息体。

## 它在流水线中的位置

### 数据集与 Collator

collator 会先把样本字段整理成统一形式，保证每个字段要么是：

* batch 维对齐的 tensor
* 可放入 `non_tensor_batch` 的 `object` numpy 数组

这是原始数据被塑造成统一协议的第一步。

### Rollout 与生成

worker 的 `generate()` 接收 prompt 侧输入的 `DataProto`，完成生成后再返回一个新的 `DataProto`。

`postprocess_generate()` 会把生成结果整理成标准训练 batch，包含如下字段：

* `prompts`
* `responses`
* `input_ids`
* `attention_mask`
* `position_ids`
* `prompt_mask`
* `response_mask`

也就是在这里，prompt 输入被转成真正的 rollout 训练数据。

### 奖励计算

reward worker 的输入输出也都是 `DataProto`，典型输出字段包括：

* `token_level_rewards`
* `response_level_rewards`
* `scores`

随后这些字段还会继续经历归一化、裁剪或 token 级展开。

### Actor 与 Critic 训练

在进入训练之前，同一个 batch 上还会逐步补充更多字段：

* `old_log_probs`
* `ref_log_probs`
* `values`
* `advantages`
* PPO 或 GAE 需要的 mask 字段

ROLL 的一个显著模式就是：它通常不会在不同阶段之间频繁切换自定义数据类，而是不断在同一个 `DataProto` 上增量补齐字段。

## RLVR 场景中的典型生命周期

在 RLVR 中，一个 batch 的生命周期通常如下：

1. collator 构造符合协议的 batch
2. actor infer 消费 prompt 侧 `DataProto`
3. 生成结果被后处理成 rollout tensor
4. reward worker 返回奖励相关字段
5. reference 和 actor train 计算 log-prob 相关字段
6. critic 在需要时补 value 字段
7. 奖励归一化、KL 惩罚和 advantage 计算继续补训练字段
8. 最终 `DataProto` 被送入 actor/critic 训练

这也是为什么理解 `DataProto` 往往是理解 RLVR pipeline 的最短路径。

## Agentic 场景中的使用方式

在 agentic 训练中，`DataProto` 还会承载轨迹级别的非 tensor 字段，例如：

* `traj_id`
* `step`
* `state_hash`
* `step_scores`

随后 pipeline 会依赖：

* `group_by()` 按轨迹或状态分组
* `reorder()` 恢复时间顺序
* `concat()` 把分组结果重新合并

这说明 `DataProto` 并不只服务于单轮文本生成，也足够适配轨迹式强化学习场景。

## 与 Dynamic Batching 的关系

dynamic batching 不仅把 `DataProto` 当作数据载体，也把它当作调度信息载体。

在分 shard 和重排之后，batch 会在 `meta_info` 中写入控制字段，例如：

* `global_micro_batch_indices`
* `global_micro_batch_lengths`
* `micro_batch_indices`
* `micro_batch_lengths`
* `num_micro_batchs`

这样，后续训练阶段不仅能拿到数据本身，还能拿到“这批数据应该如何被消费”的调度说明。

## 优点

这套设计有几个很明显的优点：

* 整个训练栈使用统一协议
* 对 tensor 和非 tensor 混合负载支持很好
* 通过 Ray 友好的序列化天然支持分布式
* 内置一组高频批处理操作
* 对新增 worker、reward 逻辑和 pipeline 比较容易扩展

## 局限与风险

当然也存在一些权衡：

* 只支持一个 batch 维
* `meta_info` 很灵活，但类型约束较弱
* 很多操作会按引用传递 `meta_info`，共享修改需要小心
* `group_by()` 用起来方便，但并不是面向超大 batch 优化的
* `to(device)` 不会自动转换 `non_tensor_batch` 中的复杂对象

实际工程里，最大的风险通常是 `meta_info` 被塞入越来越多隐式约定字段，导致排查问题时很难快速定位来源。

## 一个好用的心智模型

可以把 `DataProto` 简单理解成：

* `batch` 是 tensor 表
* `non_tensor_batch` 是样本级对象表
* `meta_info` 是批级协议头

从这个角度看，`DataProto` 就是 ROLL 在系统内部流转数据的主要中间表示。

## 总结

如果你要读懂 ROLL 的源码，`DataProto` 是非常值得优先吃透的抽象之一。只要把它的结构和语义理解清楚，数据集整理、rollout、奖励计算、分布式调度和训练更新之间的数据流转就会清晰很多。
