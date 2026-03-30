# ROLL 现有数据合成算法与代码架构调研

## 1. 文档目标

本文档面向 ROLL 仓库的开发者，梳理当前与“数据合成”直接相关的算法形态、运行链路与代码架构。

这里的“数据合成”不是狭义的离线 synthetic dataset 生成器，而是更广义的训练信号构造过程，包括：

- 基于 prompt 数据集的在线 response 合成
- 基于环境交互的多轮 trajectory 合成
- 基于 teacher 模型的 soft target / KL 信号合成
- 基于 reward、KL、advantage、mask 的训练样本再加工

ROLL 当前并没有一个单独的 `synthesis/` 模块；“数据合成”能力是分散在不同 pipeline、scheduler、worker 和 reward/advantage 后处理中实现的。

## 2. 总览结论

ROLL 当前的“数据合成”主线可以分成四类：

1. `RLVR` 在线采样
2. `Agentic` 环境 rollout
3. `Distill` 离线蒸馏
4. `On-Policy Distill (OPD)` 在线蒸馏

其中：

- `RLVR` 负责把静态 prompt 数据集转成 `prompt -> response -> reward -> advantage -> train batch`
- `Agentic` 负责把环境定义转成 `state / action / observation / reward / trajectory -> train batch`
- `Distill` 负责把 teacher 输出转成 student 的 soft supervision
- `OPD` 不单独起一套 trainer，而是复用 RLVR / Agentic pipeline，把 teacher KL 直接转成训练信号

对应入口：

- `examples/start_rlvr_pipeline.py`
- `examples/start_agentic_pipeline.py`
- `examples/start_distill_pipeline.py`
- `examples/start_onpolicy_distill_pipeline.py`

## 3. 代码架构分层

ROLL 与数据合成相关的代码，大体可以分为五层。

### 3.1 配置层

职责：

- 定义 pipeline 类型
- 定义数据源、模板、采样参数
- 定义训练/推理/reward/reference/teacher 等角色
- 定义算法开关，例如 `adv_estimator`、`pg_variant`、`use_opd`

核心目录：

- `roll/configs/base_config.py`
- `roll/configs/data_args.py`
- `roll/configs/generating_args.py`
- `roll/configs/training_args.py`
- `roll/configs/worker_config.py`

这一层决定了“什么数据被采、如何采、采完之后如何转成优化目标”。

### 3.2 数据层

职责：

- 加载本地 json/jsonl/parquet/csv/text 数据
- 将 `prompt` 或 `messages` 转成模型输入
- 提供 collator、sampler、dataset manager

核心目录：

- `roll/datasets/dataset.py`
- `roll/datasets/collator.py`
- `roll/datasets/chat_template.py`
- `roll/datasets/global_dataset.py`

这一层不负责“合成”结果本身，而是负责给生成链路提供原始输入。

### 3.3 Pipeline 层

职责：

- 串起 model update、rollout/generate、reward、advantage、train step、checkpoint
- 组织多角色 cluster 的生命周期
- 实现不同训练范式

核心目录：

- `roll/pipeline/rlvr`
- `roll/pipeline/agentic`
- `roll/pipeline/distill`
- `roll/pipeline/dpo`
- `roll/pipeline/sft`

其中与数据合成最直接相关的是 `rlvr`、`agentic`、`distill`。

### 3.4 Distributed Runtime 层

职责：

- 通过 Ray 组织分布式 worker
- 实现 generate / reward / rollout / routing / buffering / queueing
- 支撑异步采样与部分 off-policy 训练

核心目录：

- `roll/distributed/executor`
- `roll/distributed/scheduler`

这一层是 ROLL 数据合成真正的执行底座。

### 3.5 Strategy 层

职责：

- 统一训练与推理后端
- 抽象 DeepSpeed / Megatron / FSDP2 / vLLM / SGLang

核心目录：

- `roll/distributed/strategy`

这一层不直接决定“采什么数据”，但直接决定“怎样更快地产生数据与消费数据”。

## 4. 数据合成的四条主链路

## 4.1 RLVR：从静态数据集到在线训练 batch

### 4.1.1 目标

RLVR 的目标是把离线问题数据集转成可用于 RL 训练的在线样本。

最典型链路：

1. 读取 prompt/messages 数据
2. 编码成 token ids
3. 用 actor_infer 生成多个 response
4. 用 reward worker 计算分数
5. 组装 `scores / old_log_probs / ref_log_probs / values / masks`
6. 计算 token-level reward 与 advantage
7. 送入 actor_train / critic 训练

### 4.1.2 关键入口

核心实现：

- `roll/pipeline/rlvr/rlvr_pipeline.py`
- `roll/pipeline/rlvr/rlvr_config.py`
- `roll/distributed/scheduler/generate_scheduler.py`
- `roll/pipeline/rlvr/actor_pg_worker.py`
- `roll/utils/functionals.py`

入口脚本：

- `examples/start_rlvr_pipeline.py`

### 4.1.3 输入数据如何进入系统

RLVR 使用 `roll.datasets.dataset.get_dataset` 读取数据。

支持：

- 单文件
- 多文件
- 目录
- json/jsonl/parquet/csv/text

配置由 `DataArguments` 提供，关键字段包括：

- `file_name`
- `dataset_dir`
- `dataset_type`
- `prompt`
- `messages`
- `response`
- `domain_interleave_probs`

在 `rlvr_pipeline.py` 中，输入数据会先经过：

- `get_dataset(...)`
- `get_encode_function(...)`
- `preprocess_dataset(...)`

其中 `get_encode_function` 支持两种主要输入模式：

- `messages`：多轮消息结构，走 chat template
- `prompt`：直接字符串 prompt

### 4.1.4 多域数据组织方式

RLVR 把多任务训练设计成一等能力。

具体做法：

- 数据集样本可带 `tag`
- 配置中的 reward domain 通过 `tag_included` 声明自己负责哪些 tag
- `update_dataset_domain(...)` 将样本 tag 映射为 domain
- 最终形成 `self.domain_datasets: Dict[str, Dataset]`

之后根据：

- `actor_train.data_args.domain_interleave_probs`

为每个 domain 计算：

- `domain_batch_size`

然后为每个 domain 单独创建一个 `DynamicSamplingScheduler`。

这意味着 RLVR 的“数据合成”不是一锅端，而是“每个 domain 独立采样，再聚合”。

### 4.1.5 真正的数据合成发生在哪里

真正的在线合成发生在 `DynamicSamplingScheduler.get_batch(...)`。

其核心职责是：

1. 从 domain dataset 中取 prompt
2. 向 inference cluster 发 generate 请求
3. 对返回结果调用 reward 计算
4. 将结果写入 replay buffer
5. 拼出本 step 所需的 batch

这个 scheduler 支持两种模式：

- `generate_opt_level == 0`
  - 同步串行风格
  - 直接生成并算 reward
- `generate_opt_level > 0`
  - 通过 `ReplayBuffer + RouterManager + sending_request` 做异步调度

异步模式下的数据路径大致是：

1. `advance_step(...)`
2. `router_manager.resume()`
3. `replay_buffer.poll()` 取 prompt
4. `RolloutContext.process_new_prompt(...)`
5. `generate(...)`
6. `compute_rewards(...)`
7. 写回 `ReplayBuffer`
8. `replay_buffer.get_batch(...)` 收集完成样本
9. `collect_items_as_batch(...)` 合并为 `DataProto`

### 4.1.6 RLVR 支持的“合成算法”

RLVR 不是单一 GRPO 或 PPO 实现，而是一个可组合框架。

#### Advantage 估计

通过 `base_config.py` 中的 `adv_estimator` 控制，当前可见支持：

- `gae`
- `reinforce`
- `grpo`
- `gigpo`
- `step_reinforce`
- `agentic_reinforce`

其中在 RLVR 主线上常见的是：

- `gae`
- `reinforce`
- `grpo`
- `gigpo`
- `step_reinforce`

实现核心在：

- `roll/utils/functionals.py::compute_advantage`

#### Policy Gradient 变体

通过 actor worker 中的 `pg_variant` 实现，当前可见支持：

- `vanilla`
- `ppo`
- `tis`
- `topr`
- `cispo`
- `kimi15`

实现核心在：

- `roll/pipeline/rlvr/actor_pg_worker.py`

这意味着“合成出来的数据”可以被不同的策略目标消费，而不是绑定单一 loss。

#### Reward 后处理

RLVR 对 reward 的处理不是直接把分数塞进 loss，而是有完整的后处理链：

1. response-level score
2. reward normalization
3. reward clipping
4. token-level reward projection
5. KL penalty
6. advantage / return

相关实现主要在：

- `reward_postprocess(...)`
- `compute_token_reward(...)`
- `compute_advantage(...)`

位于：

- `roll/utils/functionals.py`

### 4.1.7 RLVR 中的数据合成产物

RLVR 最终产出的不是简单的 `(prompt, response)`，而是包含 RL 所需所有上下文的 `DataProto`。

典型字段包括：

- `input_ids`
- `attention_mask`
- `response_mask`
- `scores`
- `token_level_rewards`
- `advantages`
- `returns`
- `old_log_probs`
- `ref_log_probs`
- `values`
- `domain`
- `tag`
- `sample_uuid`

因此，ROLL 的 RLVR 合成本质上是“训练 batch 合成”，不是“文本样本合成”。

## 4.2 Agentic：从环境交互到 trajectory 合成

### 4.2.1 目标

Agentic pipeline 不从静态 prompt 样本直接采单轮 response，而是从环境定义出发生成多轮交互轨迹。

这类数据通常包含：

- state
- action
- observation
- step reward
- episode reward
- trajectory id
- response mask / segment 信息

因此，Agentic 的数据合成更接近“环境 rollout”。

### 4.2.2 关键入口

核心实现：

- `roll/pipeline/agentic/agentic_pipeline.py`
- `roll/pipeline/agentic/agentic_config.py`
- `roll/distributed/scheduler/rollout_scheduler.py`
- `roll/pipeline/agentic/environment_worker.py`
- `roll/pipeline/agentic/utils.py`
- `roll/pipeline/agentic/agentic_actor_pg_worker.py`

入口脚本：

- `examples/start_agentic_pipeline.py`

### 4.2.3 输入不是 dataset，而是 env config

Agentic 的训练输入由 `AgenticConfig.custom_envs` 和 `EnvManagerConfig` 定义。

关键配置包括：

- `train_env_manager`
- `val_env_manager`
- `custom_envs`
- `num_env_groups`
- `group_size`
- `group_size_redundancy`
- `max_traj_per_env`
- `tags`
- `num_groups_partition`

在 `AgenticConfig.make_env_configs(...)` 中，系统会把这些配置展开成：

- 每个 worker 管哪些 env
- 每个 env 对应什么 tag
- 每个 group 用什么 seed
- 每个 env 使用什么 `env_manager_cls`

这一步完成后，训练的原始输入已经不是普通 dataset row，而是环境实例集合。

### 4.2.4 真正的数据合成发生在哪里

Agentic 的真正合成发生在 `RolloutScheduler.get_batch(...)`。

它的核心职责是：

1. 初始化 env manager cluster
2. 更新当前全局 step
3. 恢复 router
4. 让 environment worker 持续运行 rollout loop
5. 从 `GroupQueueManager` 里按 group 拉取 trajectory batch
6. 合并为 `DataProto`

也就是说，Agentic 不是“从 dataset 取样本 -> generate”，而是：

- `EnvironmentWorker.run_rollout_loop(...)`
- 通过 inference cluster 与环境交互
- 把 rollout 结果写入 `GroupQueueManager`
- 由 `RolloutScheduler` 统一收集

### 4.2.5 Group 机制的意义

Agentic 里有一套很强的 group 设计：

- 同一组环境共享 seed / 配置
- `rollout_batch_size` 必须能被 `group_size` 整除
- scheduler 以 group 为单位收集数据

这个设计主要服务于：

- 方差降低
- 多样性受控比较
- stepwise / trajwise 算法需求

因此，Agentic 的“数据合成单元”很多时候不是单条样本，而是一个 group 下的多条轨迹。

### 4.2.6 Agentic 支持的“合成算法”

Agentic 的 reward 与 advantage 处理与 RLVR 有共性，但多了 trajectory / step 维度。

当前核心支持：

- `gigpo`
- `step_reinforce`
- `agentic_reinforce`
- `gae`
- `reinforce`
- `grpo`

其中：

- `gigpo` 偏 step-wise / group-wise 学习
- `agentic_reinforce` 更强调 agent 轨迹结构
- `ratio_type` 还支持 `token` / `segment`

相关实现主要在：

- `roll/pipeline/agentic/utils.py`
- `roll/pipeline/agentic/agentic_actor_pg_worker.py`

### 4.2.7 Agentic 中的 reward 合成

Agentic 的 reward 不一定是单个 response score。

系统内部会根据配置组合出：

- response-level rewards
- episode rewards
- step rewards
- discounted returns
- KL penalty

关键函数包括：

- `compute_response_level_rewards(...)`
- `compute_discounted_returns(...)`
- `agentic_compute_advantage(...)`

因此 Agentic 的“数据合成”不仅合成文本，还会合成：

- 段级 reward
- step 级回报
- segment mask
- trajectory 级统计量

### 4.2.8 Agentic 中的数据合成产物

最终训练 batch 常常包含：

- trajectory 相关 id
- 多轮 response mask
- step / segment 对齐信息
- reward 与 discounted return
- old/ref log probs
- 可能的 value estimates

相比 RLVR，它的 batch 结构更复杂，也更依赖环境协议。

## 4.3 Distill：teacher logits 驱动的离线监督信号合成

### 4.3.1 目标

Distill pipeline 的目标是：

- 从已有监督数据出发
- 让 teacher 产生 soft target
- 将 soft target 通过 logits transfer 送给 student
- 用 distill loss + SFT loss 联合训练

这条链路不做 RL rollout，但也是数据合成，因为 teacher 输出本身就是一种动态监督信号。

### 4.3.2 关键入口

核心实现：

- `roll/pipeline/distill/distill_pipeline.py`
- `roll/pipeline/distill/distill_config.py`
- `roll/pipeline/distill/distill_worker.py`
- `roll/pipeline/distill/logits_transfer_group.py`
- `roll/pipeline/distill/various_divergence.py`

入口脚本：

- `examples/start_distill_pipeline.py`

### 4.3.3 数据进入方式

Distill 直接读取 json 数据集，然后通过 `preprocess_dataset(...)` 转成：

- `input_ids`
- `attention_mask`
- `labels`

其编码逻辑支持：

- `prompt_key`
- `question_key`
- `answer_key`
- `system_key`
- `distill_on_prompt`

这意味着它支持：

- 只蒸馏 answer 部分
- 或对整段 prompt + answer 进行蒸馏

### 4.3.4 真正的“合成”发生在哪里

Distill 的合成不是文本生成，而是 teacher 前向得到的 soft target。

主要步骤：

1. teacher 对 batch 前向
2. teacher logits 通过 `LogitsTransferGroup` 传递
3. student 读取 teacher probs / logits
4. 计算 divergence-based distill loss
5. 与 GPT/SFT loss 按比例混合

关键公式上的控制参数：

- `top_k_logits`
- `distill_loss_weight`
- `distill_on_prompt`
- `VariousDivergence`

这一套机制的本质是“在线合成软标签”。

### 4.3.5 Distill 的产物

Distill 最终产出的不是 rollout batch，而是：

- teacher-forward 产生的 soft target
- student 可消费的蒸馏监督

因此这条链路偏“监督信号合成”。

## 4.4 On-Policy Distill：复用 RL pipeline 的在线蒸馏

### 4.4.1 关键结论

OPD 在当前仓库中不是单独的 `roll/pipeline/onpolicy_distill/` 目录，而是通过配置复用现有 RL pipeline。

入口脚本：

- `examples/start_onpolicy_distill_pipeline.py`

它根据配置字段：

- `pure_opd_pipeline_type`

动态选择：

- `RLVRConfig + RLVRPipeline`
- 或 `AgenticConfig + AgenticPipeline`

### 4.4.2 角色映射方式

在 `base_config.py::_handle_opd_mapping()` 中：

纯 OPD 模式：

- `student_train -> actor_train`
- `student_infer -> actor_infer`
- `teacher -> reference`

混合 OPD 模式：

- `teacher -> reference`

也就是说，OPD 没有重新实现 rollout，只是重写了角色语义。

### 4.4.3 奖励如何合成

在 `compute_advantage(...)` 中：

- 如果 `is_pure_opd=True`
  - advantage 直接取 `-KL(student || teacher)` 形式
- 如果 `use_opd=True`
  - 则在普通 RL advantage 上减去 `opd_kl_coef * kld`

因此 OPD 的关键不是新 scheduler，而是把 teacher KL 注入到 reward/advantage 语义里。

### 4.4.4 OPD 的意义

它把“teacher 监督”从离线 distill 扩展成在线 rollout 场景下的训练信号。

从系统设计上看，这是一个非常克制的实现：

- 不增加新运行时层
- 不复制 RL pipeline
- 只复用 `reference` 通路与 advantage 逻辑

## 5. Runtime 视角下的核心执行组件

## 5.1 Cluster

`roll.distributed.executor.cluster.Cluster`

职责：

- 管理一组同角色 worker
- 负责 initialize / load_states / offload_states / train_step / generate / compute_log_probs

在所有合成链路里，Cluster 都是角色容器。

常见角色：

- `actor_train`
- `actor_infer`
- `reference`
- `critic`
- `reward`
- `teacher`
- `student`
- `env_manager`

## 5.2 RouterManager

`roll.distributed.scheduler.router.RouterManager`

职责：

- 管理请求路由
- 将 generate 请求发给具体 infer worker
- 控制 suspend / resume / abort

它是在线合成吞吐的关键控制点。

在 RLVR 中，它主要服务 prompt 生成。

在 Agentic 中，它主要服务 environment rollout 期间的模型调用。

## 5.3 DynamicSamplingScheduler

`roll.distributed.scheduler.generate_scheduler.DynamicSamplingScheduler`

这是 RLVR 数据合成最核心的 scheduler。

职责：

- 遍历 dataset
- 管理 prompt sampling
- 驱动 generate
- 驱动 reward
- 管理 replay buffer
- 聚合为 batch

如果后续想加新的 prompt 级在线合成算法，这里是最重要的扩展点。

## 5.4 RolloutScheduler

`roll.distributed.scheduler.rollout_scheduler.RolloutScheduler`

这是 Agentic 轨迹合成最核心的 scheduler。

职责：

- 初始化 env manager cluster
- 控制 rollout loop 生命周期
- 从 `GroupQueueManager` 收集批量轨迹
- 支持 suspend/resume
- 支持与 partial GPU overlap 协同

如果后续想做新的 environment-level 合成策略，这里是核心切入点。

## 5.5 RewardScheduler

`roll.distributed.scheduler.reward_scheduler.RewardScheduler`

职责：

- 调用 reward worker
- 将生成结果映射为 reward batch

它决定了“生成结果如何变成训练信号”。

## 5.6 DataProto

`roll.distributed.scheduler.protocol.DataProto`

这是系统中最关键的数据载体。

它承接：

- 原始 prompt batch
- 生成输出
- reward 输出
- log probs
- values
- masks
- metrics

几乎所有“数据合成”最后都会归结为对 `DataProto` 的构造、合并、切分、排序与标注。

## 6. 算法视角下的训练信号合成

从算法角度看，ROLL 合成训练信号通常经过以下步骤：

1. 原始输入构造
2. 模型生成或环境交互
3. reward 计算
4. reward 后处理
5. KL 约束注入
6. advantage / return 计算
7. policy / value loss 计算

这里最关键的不是“采样”本身，而是 3 到 6。

## 6.1 Reward 合成

ROLL 当前可见支持：

- rule-based reward
- code sandbox reward
- LLM-as-judge reward
- response length penalty
- environment-derived reward
- teacher-KL derived reward

这意味着 reward 本身已经是“第一层合成”。

## 6.2 Token-level reward 合成

很多 reward 原本是 response-level 或 trajectory-level。

ROLL 会通过函数将其投影到 token 级：

- `compute_token_reward(...)`

之后再参与 RL 训练。

因此，token-level reward 是“第二层合成”。

## 6.3 Advantage 合成

ROLL 将 token reward、value、KL 和 mask 再组合成：

- `advantages`
- `returns`

这一步由：

- `compute_advantage(...)`
- Agentic 版本的 advantage 工具

完成。

这可以看作“第三层合成”。

## 6.4 最终 loss 合成

在 actor worker 中，最终损失可进一步组合：

- PG/PPO/TOPR 等 policy loss
- KL loss
- entropy regularization
- train-infer correction

这一步才真正形成最终优化目标。

## 7. 当前架构的几个关键特点

## 7.1 没有单独的 synthesis 子系统

这既是优点也是代价。

优点：

- 不会为每种训练范式重复实现一套系统
- 共享 runtime、cluster、scheduler、DataProto

代价：

- 数据合成逻辑分散在 pipeline、scheduler、worker、utils 中
- 首次阅读成本较高

## 7.2 训练时在线合成为主，不是离线预生成

当前主流路径：

- RLVR：在线 generate + reward
- Agentic：在线 rollout
- OPD：在线 KL 蒸馏

只有 Distill 更偏离线监督数据驱动。

这说明 ROLL 的设计中心是“训练时构造样本”，不是“先生成语料再微调”。

## 7.3 Scheduler 是复杂度核心

模型本身的训练逻辑相对常规，真正复杂的是：

- prompt 如何发
- 请求如何路由
- reward 如何异步回来
- 何时暂停与恢复
- 如何兼容 off-policy / async_generation_ratio
- 如何在 partial GPU overlap 下 shrink / expand sampler

所以二次开发时，很多性能或合成策略问题最终都会落在 scheduler。

## 7.4 OPD 的设计非常克制

OPD 没有单独复制一套 distill runtime，而是：

- 复用 RLVR / Agentic pipeline
- 复用 reference 通道
- 只在 config mapping 和 advantage 计算处改语义

这使它非常适合继续扩展。

## 8. 推荐阅读顺序

如果目标是快速理解 ROLL 数据合成系统，建议按下面顺序阅读。

### 8.1 第一层：入口与配置

先读：

- `examples/start_rlvr_pipeline.py`
- `examples/start_agentic_pipeline.py`
- `examples/start_distill_pipeline.py`
- `examples/start_onpolicy_distill_pipeline.py`
- `roll/configs/base_config.py`
- `roll/pipeline/rlvr/rlvr_config.py`
- `roll/pipeline/agentic/agentic_config.py`

目标：

- 理解系统有哪些 pipeline
- 理解有哪些角色
- 理解算法开关有哪些

### 8.2 第二层：主 pipeline

再读：

- `roll/pipeline/rlvr/rlvr_pipeline.py`
- `roll/pipeline/agentic/agentic_pipeline.py`
- `roll/pipeline/distill/distill_pipeline.py`

目标：

- 理解一轮 step 如何组织
- 理解 rollout / reward / train 的阶段边界

### 8.3 第三层：scheduler

重点读：

- `roll/distributed/scheduler/generate_scheduler.py`
- `roll/distributed/scheduler/rollout_scheduler.py`
- `roll/distributed/scheduler/router.py`
- `roll/distributed/scheduler/reward_scheduler.py`

目标：

- 理解数据到底如何被合成出来
- 理解异步与路由控制

### 8.4 第四层：算法处理

最后读：

- `roll/utils/functionals.py`
- `roll/pipeline/agentic/utils.py`
- `roll/pipeline/rlvr/actor_pg_worker.py`
- `roll/pipeline/agentic/agentic_actor_pg_worker.py`
- `roll/pipeline/distill/distill_worker.py`

目标：

- 理解 reward / advantage / loss 的构造

## 9. 如果要做二次开发，推荐的扩展点

### 9.1 新的数据源

改动位置：

- `roll/datasets/dataset.py`
- `roll/configs/data_args.py`

适合：

- 新文件格式
- 新远程数据源
- 新的输入字段约定

### 9.2 新的 prompt 级在线合成策略

改动位置：

- `roll/distributed/scheduler/generate_scheduler.py`
- `roll/distributed/scheduler/user_defined_rollout_loop.py`

适合：

- 新采样策略
- 多次重试
- 动态扩样
- response 过滤策略

### 9.3 新的 environment / trajectory 合成策略

改动位置：

- `roll/distributed/scheduler/rollout_scheduler.py`
- `roll/pipeline/agentic/environment_worker.py`
- 对应 env manager

适合：

- 新环境协议
- 新 trajectory 聚合方式
- 新 step-wise 训练范式

### 9.4 新 reward 体系

改动位置：

- `roll/pipeline/rlvr/rlvr_config.py`
- reward worker 实现
- `roll/distributed/scheduler/reward_scheduler.py`

适合：

- rule reward
- LLM judge
- 多 reward 融合
- reward filtering

### 9.5 新 advantage / loss 设计

改动位置：

- `roll/utils/functionals.py`
- `roll/pipeline/agentic/utils.py`
- `roll/pipeline/rlvr/actor_pg_worker.py`
- `roll/pipeline/agentic/agentic_actor_pg_worker.py`

适合：

- 新 advantage estimator
- 新 PG 变体
- 新 KL 组合方式

## 10. 最终总结

ROLL 当前的“数据合成系统”本质上是一个分布式在线训练样本构造框架，而不是单独的 synthetic data 工具。

它的核心特点是：

- 以 pipeline 为上层编排单元
- 以 scheduler 为在线合成执行核心
- 以 `DataProto` 为统一数据载体
- 以 reward/KL/advantage/loss 后处理为训练信号合成核心

从实现形态看：

- `RLVR` 负责 prompt 到 response 的在线样本合成
- `Agentic` 负责环境交互到 trajectory 的在线样本合成
- `Distill` 负责 teacher soft target 的监督信号合成
- `OPD` 负责把 teacher KL 注入现有 RL pipeline，形成在线蒸馏信号

如果只用一句话概括：

ROLL 不是“先有数据、再训练”的框架，而是“在训练循环内部持续合成数据与训练信号”的框架。
