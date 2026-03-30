# ROLL 后端架构方案

**文档时间**: 2026-03-30
**适用对象**: ROLL 控制面、训练平台、内部研发与运维团队
**设计基线**: 基于当前 ROLL 仓库真实结构，围绕 `ROLL + Ray` 构建后端控制面，而非重写训练内核

---

## 摘要

基于当前 `ROLL` 的代码结构和能力边界，后端不应直接承载训练主逻辑，而应作为 `ROLL + Ray` 之上的控制面。`ROLL` 继续负责 pipeline 执行、Ray 资源调度、模型训练/推理角色协同；新后端负责作业编排、配置模板化、运行状态管理、日志指标聚合、鉴权与审计、对外 API。

主框架建议固定为 **FastAPI**，不建议用 Flask 作为主实现。原因是 ROLL 的后端天然需要异步任务提交、SSE/WebSocket 日志流、Pydantic 配置校验、OpenAPI 文档、后台任务编排，以及较强的类型约束，FastAPI 更适合这类训练控制面场景。

推荐目标形态为：

1. **v1 单集群控制面**: 一个 FastAPI 服务对接一个 ROLL/Ray 集群，先把作业管理、配置管理和观测做好。
2. **v2 平台化演进**: 在 API 保持兼容的前提下，上提集群抽象、队列调度和资源池管理能力，支持多集群。

---

## 一、设计原则

### 1.1 保持训练面与控制面解耦

- `ROLL` 负责训练、推理、分布式执行和资源编排
- 新后端负责任务提交、配置治理、运行管理、审计和观测
- 后端不侵入 `roll/pipeline` 核心逻辑，不把训练逻辑重写成 Web Handler

### 1.2 以配置驱动为中心

当前 ROLL 是典型的 Hydra/YAML 驱动框架，已有大量 `examples/start_*_pipeline.py` 启动入口和成体系的 dataclass 配置定义。因此后端必须围绕“模板 + 参数覆盖 + 配置快照 + 版本化”设计，而不是只提供一层薄薄的命令包装。

### 1.3 先单体控制面，后平台化

v1 不建议过早拆成微服务。单体控制面更适合快速落地、调试简单、问题边界清晰。只要提前设计清楚执行器接口、元数据模型和事件流，后续演进到多集群时不需要推翻重来。

### 1.4 强调可观测性和可审计性

ROLL 的核心场景是长时间运行的大规模训练与推理任务。后端设计必须把日志、指标、checkpoint 索引、运行状态、操作审计视为一等公民，而不是事后补充。

---

## 二、结合 ROLL 现状的架构判断

从仓库结构可以确认以下事实：

- `roll/pipeline/` 下已有 `RLVRPipeline`、`AgenticPipeline`、`DistillPipeline`、`DPOPipeline`、`SFTPipeline`、`RewardFLPipeline`
- 各类 pipeline 通过 `examples/start_*_pipeline.py` 入口启动
- 配置通过 Hydra 和 dataclass 体系管理，如 `BaseConfig`、`RLVRConfig`、`AgenticConfig`
- `roll/distributed/scheduler/initialize.py` 明确负责 Ray cluster 初始化
- `BasePipeline` 已具备 tracker、checkpoint、状态持久化等能力

因此，后端最合理的定位不是“新的训练框架”，而是：

> **围绕 ROLL pipeline 的统一控制面与作业编排层**

这意味着后端的职责是：

- 管理任务生命周期
- 管理模板与配置
- 渲染最终 Hydra 配置
- 调起现有 ROLL 启动入口
- 聚合日志、指标和产物
- 为前端或平台提供统一 API

---

## 三、推荐技术栈

### 3.1 主框架

- **FastAPI**

原因：

- 原生支持异步 I/O
- 与 Pydantic v2 配合适合复杂配置校验
- 自动生成 OpenAPI 文档
- 适合做实时日志流、事件流和控制面 API
- 比 Flask 更适合中大型、强类型、异步化控制面

### 3.2 数据与缓存

- **PostgreSQL**: 元数据主存储
- **Redis**: 队列、缓存、日志流、分布式锁

不建议 v1 使用 MySQL 作为主库，原因不是不能用，而是 PostgreSQL 在 JSONB、复杂查询、审计和事件类元数据场景里更合适。

### 3.3 ORM 与模型层

- **SQLAlchemy 2.0**
- **Alembic**
- **Pydantic v2**

### 3.4 后台任务

- **Celery + Redis** 或 **RQ + Redis**

推荐 v1 直接用 `Celery + Redis`：

- 任务状态管理成熟
- 失败重试和路由机制更完整
- 与 FastAPI 配合方式清晰

### 3.5 可观测性

- **Prometheus**: 指标采集
- **Grafana**: 指标展示
- **Loki** 或 **ELK**: 日志聚合
- 同时保留对 `TensorBoard / WandB / SwanLab` 的外链聚合能力

### 3.6 对象存储

- **MinIO / S3 Compatible Storage**

用于存放：

- 渲染后的配置快照
- 日志归档
- checkpoint 索引
- 导出报告

---

## 四、总体架构

### 4.1 分层结构

```text
Frontend / CLI / OpenAPI Clients
            |
            v
    FastAPI Control Plane
            |
            +-------------------+
            |                   |
            v                   v
   Application Services     Auth / RBAC / Audit
            |
            v
     Execution Adapter
            |
            v
   Worker Queue + Executors
            |
            v
   ROLL Entrypoints + Ray Cluster
            |
            v
   Training / Inference / Checkpoints / Trackers
```

### 4.2 模块职责

- **API Gateway / Control Plane**
  - 提供 REST API、SSE、WebSocket、鉴权、配置校验、OpenAPI 文档
- **Application Service**
  - 负责任务编排、状态机、模板管理、权限检查、审计落库
- **Execution Adapter**
  - 负责把标准化任务请求转换成 ROLL 启动命令
- **Executor Worker**
  - 负责拉起训练进程、消费日志、更新状态、处理失败重试
- **Metadata Store**
  - 保存用户、角色、项目、模板、任务、运行、产物、审计信息
- **Artifact Store**
  - 保存配置快照、日志归档、checkpoint 索引

---

## 五、与 ROLL 的职责边界

### 5.1 ROLL 负责

- pipeline 实例化
- Ray 集群初始化
- actor / infer / reward / env manager 的资源协同
- 训练循环与 checkpoint
- 训练指标追踪

### 5.2 后端负责

- 将“用户任务”转换成“标准化 pipeline 执行请求”
- 管理 Job / Run 生命周期
- 管理模板、模型引用、数据集引用
- 聚合 tracker 链接、日志、checkpoint 索引
- 提供停止、重试、恢复等控制能力
- 记录操作审计与权限控制

### 5.3 不建议的做法

- 不要把 pipeline 直接嵌进 FastAPI 请求线程
- 不要在 API 进程内部持有 Ray 训练生命周期
- 不要允许所有用户任意上传 YAML 并原样执行

---

## 六、服务内部模块设计

建议新增一个独立后端包，例如：

```text
roll/server/
├── api/
│   └── v1/
│       ├── endpoints/
│       │   ├── auth.py
│       │   ├── jobs.py
│       │   ├── runs.py
│       │   ├── templates.py
│       │   ├── projects.py
│       │   ├── clusters.py
│       │   ├── artifacts.py
│       │   └── metrics.py
│       └── router.py
├── core/
│   ├── config.py
│   ├── security.py
│   ├── db.py
│   ├── logging.py
│   └── dependencies.py
├── schemas/
│   ├── auth.py
│   ├── jobs.py
│   ├── runs.py
│   ├── templates.py
│   └── common.py
├── domain/
│   ├── enums.py
│   ├── state_machine.py
│   └── permissions.py
├── repositories/
│   ├── jobs.py
│   ├── runs.py
│   ├── templates.py
│   └── artifacts.py
├── services/
│   ├── auth_service.py
│   ├── job_service.py
│   ├── run_service.py
│   ├── template_service.py
│   ├── artifact_service.py
│   └── cluster_service.py
├── executors/
│   ├── base.py
│   ├── local_process_executor.py
│   └── ray_job_executor.py
├── workers/
│   ├── tasks.py
│   └── log_consumer.py
├── models/
│   └── orm.py
├── app.py
└── main.py
```

这个结构与当前 `docs_roll/backend_api.md` 中的方向一致，但更完整，适合正式落地。

---

## 七、领域模型设计

### 7.1 核心实体

#### User

- 用户基础信息
- 登录信息
- 所属项目和角色

#### Role

- `admin`
- `operator`
- `viewer`

#### Project

- 逻辑工作空间
- 隔离任务、模板、数据集引用、模型引用

#### Cluster

即使 v1 只有一个集群，也保留该实体。

建议默认创建：

- `default-cluster`

#### PipelineTemplate

模板实体，用于描述：

- pipeline 类型
- 默认参数
- 可覆盖字段白名单
- 资源画像
- 启动入口

#### DatasetRef

记录：

- 数据集名称
- 路径
- 版本
- 格式说明
- 数据来源

#### ModelRef

记录：

- 模型名称
- 路径或远程引用
- 模型类型
- 适用 pipeline

#### Job

逻辑任务定义。

一个 Job 可以重试多次，形成多个 Run。

#### Run

任务的一次具体执行实例，带有：

- 运行状态
- 启动时间 / 结束时间
- 退出码
- 配置快照
- 日志索引
- 指标摘要

#### Artifact

记录配置快照、日志文件、checkpoint 索引、导出报告等。

#### AuditLog

记录所有敏感操作。

---

## 八、任务与运行模型

### 8.1 Job 与 Run 拆分

建议明确区分：

- **Job**: 用户定义的逻辑任务
- **Run**: Job 的一次执行实例

示例：

- 用户创建一个 RLVR 训练任务，对应一个 Job
- 第一次执行失败，对应 Run #1
- 用户点击重试，生成 Run #2

这样可以保留完整运行历史和失败上下文。

### 8.2 Run 状态机

建议固定为以下状态：

- `pending`
- `validating`
- `queued`
- `starting`
- `running`
- `stopping`
- `succeeded`
- `failed`
- `canceled`

状态切换必须由服务端统一控制，不允许客户端任意写入。

### 8.3 幂等性

对创建 Run 的请求引入幂等键，避免前端重复点击导致重复执行。

---

## 九、Pipeline 抽象与模板体系

### 9.1 支持的 pipeline 类型

后端统一抽象以下 ROLL pipeline：

- `rlvr`
- `agentic`
- `distill`
- `dpo`
- `sft`
- `reward_fl`

### 9.2 模板化提交

主提交流程不建议允许任意 YAML 原样透传。推荐采用：

- 模板固定默认值
- 用户只覆盖允许字段
- 后端渲染最终 YAML
- 最终配置作为 artifact 固化

### 9.3 模板字段结构

每个模板至少包含：

- `pipeline_type`
- `entrypoint`
- `base_config`
- `override_whitelist`
- `resource_profile`
- `default_tracker`

### 9.4 参数覆盖策略

用户提交时可以带：

- `overrides: dict[str, Any]`

但只允许覆盖模板白名单中的字段，例如：

- `rollout_batch_size`
- `max_steps`
- `save_steps`
- `tracker_kwargs`
- `output_dir`

高风险字段需要管理员权限，例如：

- 自定义 `worker_cls`
- 任意命令行参数
- 任意路径透传
- 任意 Python 类路径

---

## 十、公共 API 设计

### 10.1 认证与权限

- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`

### 10.2 项目与成员

- `GET /api/v1/projects`
- `POST /api/v1/projects`
- `POST /api/v1/projects/{project_id}/members`

### 10.3 模板管理

- `POST /api/v1/templates`
- `GET /api/v1/templates`
- `GET /api/v1/templates/{template_id}`
- `POST /api/v1/templates/{template_id}/validate`

### 10.4 配置渲染

- `POST /api/v1/configs/render`
- `POST /api/v1/configs/diff`

### 10.5 作业管理

- `POST /api/v1/jobs`
- `GET /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `POST /api/v1/jobs/{job_id}/runs`

### 10.6 运行管理

- `GET /api/v1/runs/{run_id}`
- `POST /api/v1/runs/{run_id}/stop`
- `POST /api/v1/runs/{run_id}/retry`
- `POST /api/v1/runs/{run_id}/resume`
- `GET /api/v1/runs/{run_id}/logs`
- `GET /api/v1/runs/{run_id}/events`
- `GET /api/v1/runs/{run_id}/artifacts`

### 10.7 资源与健康

- `GET /api/v1/clusters/default/health`
- `GET /api/v1/clusters/default/resources`
- `GET /api/v1/clusters/default/ray`

### 10.8 实时接口

- `GET /api/v1/runs/{run_id}/logs/stream`
- `GET /api/v1/runs/{run_id}/metrics/stream`
- `WS /api/v1/runs/{run_id}/ws`

v1 优先使用 **SSE** 做日志和指标流，简单且稳定。WebSocket 作为补充接口即可。

---

## 十一、关键请求与响应模型

### 11.1 JobCreateRequest

至少包含：

```json
{
  "project_id": "proj_123",
  "pipeline_type": "rlvr",
  "template_id": "tpl_rlvr_default",
  "name": "qwen3-8b-rlvr-train",
  "cluster_id": "default-cluster",
  "dataset_ref_id": "dataset_math_v1",
  "model_ref_id": "model_qwen3_8b",
  "overrides": {
    "max_steps": 1000,
    "rollout_batch_size": 128
  },
  "tags": ["rlvr", "math"]
}
```

### 11.2 RunDetailResponse

至少返回：

```json
{
  "run_id": "run_001",
  "job_id": "job_001",
  "status": "running",
  "pipeline_type": "rlvr",
  "cluster_id": "default-cluster",
  "started_at": "2026-03-30T12:00:00Z",
  "finished_at": null,
  "exit_code": null,
  "config_artifact_url": "/api/v1/artifacts/art_cfg_001/download",
  "stdout_log_url": "/api/v1/runs/run_001/logs?stream=stdout",
  "stderr_log_url": "/api/v1/runs/run_001/logs?stream=stderr",
  "tracker_links": {
    "tensorboard": "http://tensorboard.local/exp/001"
  },
  "latest_metrics": {
    "system/step": 120,
    "train/reward_mean": 0.73
  }
}
```

---

## 十二、执行链路设计

### 12.1 提交流程

1. 前端选择 pipeline 模板并填写参数
2. FastAPI 进行参数校验与 RBAC 校验
3. 服务端创建 `Job` 和初始 `Run`
4. 渲染最终 Hydra/YAML 配置
5. 将配置快照保存为 artifact
6. 投递后台执行任务到 Redis 队列
7. Worker 拉起训练进程
8. 执行器实时采集 stdout/stderr 和状态
9. 训练结束后写入退出码、最终状态、产物索引、摘要指标

### 12.2 启动模式

v1 直接复用现有入口：

- `python examples/start_rlvr_pipeline.py --config_path <dir> --config_name <name>`
- `python examples/start_agentic_pipeline.py --config_path <dir> --config_name <name>`
- 其他 pipeline 同理

建议 v1.1 补充统一启动入口：

```bash
python -m roll.server.runner --pipeline rlvr --config /path/to/config.yaml
```

这样后端后续可以不再依赖 `examples/` 的脚本命名。

### 12.3 为什么不直接在 API 进程里运行

如果把训练任务直接运行在 FastAPI 进程内，会带来明显问题：

- API 进程阻塞或资源污染
- Ray 生命周期难以管理
- 请求超时与训练生命周期不一致
- 训练失败会影响整个服务稳定性

因此推荐采用：

- FastAPI 控制面
- Celery worker 执行器
- 独立子进程承载训练任务

---

## 十三、执行器设计

### 13.1 执行器接口

定义统一执行器接口：

- `prepare(run)`
- `start(run)`
- `stop(run)`
- `resume(run)`
- `collect_logs(run)`
- `collect_metrics(run)`

### 13.2 v1 执行器实现

建议优先实现：

- `LocalProcessExecutor`

职责：

- 生成临时运行目录
- 输出最终配置文件
- 注入环境变量
- 构建启动命令
- 启动子进程
- 流式读取 stdout/stderr
- 更新运行状态

### 13.3 v2 预留实现

- `RayJobExecutor`
- `K8sJobExecutor`

只要执行器接口提前抽象好，未来切换执行介质时无需改动上层 API。

---

## 十四、数据库设计

### 14.1 建议的 PostgreSQL 表

- `users`
- `roles`
- `project_members`
- `projects`
- `clusters`
- `pipeline_templates`
- `dataset_refs`
- `model_refs`
- `jobs`
- `job_runs`
- `run_events`
- `artifacts`
- `audit_logs`

### 14.2 字段重点

#### jobs

- `id`
- `project_id`
- `pipeline_type`
- `template_id`
- `name`
- `status_summary`
- `created_by`
- `created_at`

#### job_runs

- `id`
- `job_id`
- `cluster_id`
- `status`
- `attempt_no`
- `config_snapshot_path`
- `pid`
- `queue_name`
- `exit_code`
- `started_at`
- `finished_at`

#### artifacts

- `id`
- `run_id`
- `artifact_type`
- `storage_uri`
- `metadata`
- `created_at`

#### audit_logs

- `id`
- `user_id`
- `action`
- `resource_type`
- `resource_id`
- `request_id`
- `payload`
- `created_at`

### 14.3 为什么需要 JSONB

以下信息适合放在 JSONB：

- 配置快照摘要
- tracker 外链
- 指标摘要
- 审计 payload
- 扩展属性

这也是优先 PostgreSQL 的原因之一。

---

## 十五、Redis 设计

Redis 在 v1 中主要承担以下角色：

- 任务队列
- 日志流缓存
- SSE 消费游标
- 运行态缓存
- 幂等键
- 分布式锁

建议命名示例：

- `queue:runs:default`
- `stream:logs:{run_id}`
- `cache:run:{run_id}`
- `idempotency:create_run:{key}`

---

## 十六、对象存储与产物管理

建议对象存储目录结构：

```text
artifacts/
  configs/{run_id}/final.yaml
  logs/{run_id}/stdout.log
  logs/{run_id}/stderr.log
  checkpoints/{run_id}/index.json
  exports/{run_id}/summary.json
```

说明：

- 后端只维护 checkpoint 索引，不直接接管大模型权重文件生命周期
- 大文件产物继续以 ROLL 原生输出目录为准
- 后端负责把可查询索引、下载地址和摘要信息标准化

---

## 十七、日志与指标体系

### 17.1 日志来源

主要有三类：

- 子进程 stdout/stderr
- ROLL pipeline 结构化日志
- Ray 节点 / worker 日志

### 17.2 v1 做法

- 子进程日志实时写本地文件
- 同步写 Redis Stream，供 SSE 消费
- 训练结束后归档到对象存储

### 17.3 指标来源

指标分三层：

- **系统层**
  - CPU
  - 内存
  - GPU 利用率
  - 磁盘和网络
- **集群层**
  - Ray 节点状态
  - 可用资源
  - 排队任务数
- **训练层**
  - step
  - loss
  - reward
  - tokens/s
  - samples/s
  - checkpoint 次数
  - eval 结果

### 17.4 展示策略

后端自身导出 Prometheus 指标，同时在 Run 详情中聚合以下外链：

- TensorBoard
- WandB
- SwanLab

---

## 十八、权限与安全设计

### 18.1 v1 权限模型

采用基础 RBAC：

- `admin`
  - 管理用户、模板、项目、集群和全部任务
- `operator`
  - 创建、停止、重试项目内任务
- `viewer`
  - 只读查看任务、日志摘要、指标和产物索引

### 18.2 认证方式

v1 建议：

- 用户名密码登录
- JWT 访问令牌

如果未来接企业统一登录，预留 OIDC 接口即可。

### 18.3 敏感信息处理

以下信息不应明文直接暴露：

- 对象存储密钥
- 第三方 API Key
- tracker token
- judge model key

最少要求：

- 服务端加密存储
- API 返回时脱敏
- 下载配置快照时对敏感字段打码

### 18.4 审计范围

必须审计的操作：

- 登录
- 创建模板
- 修改模板
- 提交任务
- 停止任务
- 重试任务
- 恢复任务
- 下载配置快照
- 查看敏感日志

---

## 十九、部署方案

### 19.1 v1 部署组件

推荐最小部署单元：

- `backend-api`
- `backend-worker`
- `postgres`
- `redis`

外挂依赖：

- `ray head/worker`
- `ROLL training runtime`
- `minio`
- `prometheus`
- `grafana`
- `loki`

### 19.2 部署环境建议

- 开发环境：`docker-compose`
- 生产环境：`Kubernetes`

### 19.3 与 Ray/ROLL 的集成

v1 采用松耦合方式：

- 控制面不直接托管 Ray 生命周期
- 执行器所在节点需要访问训练环境
- 需要共享以下目录或等价存储：
  - 配置渲染目录
  - 输出目录
  - 日志目录
  - checkpoint 索引目录

---

## 二十、两阶段路线图

### 20.1 阶段一：可落地 v1

目标是快速形成稳定控制面。

核心范围：

- `FastAPI + PostgreSQL + Redis + Celery`
- 单集群 `default-cluster`
- 基础 RBAC
- 6 类 pipeline 模板化提交
- 任务创建、停止、重试、日志流、配置快照、artifact 查询
- 对 TensorBoard / WandB / SwanLab 做统一外链聚合

### 20.2 阶段二：平台化 v2

在不破坏 v1 API 的前提下扩展：

- 多集群管理
- 队列和优先级调度
- 项目级资源限额
- 资源配额与并发控制
- 更细粒度权限控制
- 多执行器支持：
  - `LocalProcessExecutor`
  - `RayJobExecutor`
  - `K8sJobExecutor`
- 工作流编排：
  - 训练前校验
  - 训练
  - 评估
  - 导出
  - 发布

---

## 二十一、测试与验收

### 21.1 单元测试

- 模板渲染正确
- 字段白名单覆盖生效
- 非法覆盖被拒绝
- 状态机流转合法
- RBAC 权限校验正确

### 21.2 集成测试

- 提交 RLVR 任务后生成 Job 和 Run
- 渲染配置成功并生成 artifact
- 执行器正确拉起训练子进程
- 子进程失败后 Run 标记为 `failed`
- 停止任务后状态转为 `canceled`
- 重试产生新的 Run，不污染历史 Run
- SSE 日志流可持续消费

### 21.3 验收场景

- 能成功基于 `examples/start_rlvr_pipeline.py` 提交并运行最小 RLVR 任务
- 能成功基于 `examples/start_agentic_pipeline.py` 提交并运行最小 Agentic 任务
- 管理员可创建模板
- operator 可提交和停止任务
- viewer 只能查看
- 前端可查看作业列表、运行详情、日志流、配置快照、基础指标

---

## 二十二、为什么优先 FastAPI 而不是 Flask

如果只是简单包装几个接口，Flask 可以完成任务。但结合 ROLL 的真实需求，FastAPI 更优：

- 需要异步化日志流和实时状态推送
- 需要复杂请求模型和严格字段校验
- 需要自动生成 API 文档
- 需要较多长生命周期、非同步接口
- 需要后续更平滑地引入 SSE、WebSocket、后台任务和类型约束

因此：

- **FastAPI 作为主框架**
- Flask 不作为推荐主实现

---

## 二十三、实施优先级建议

建议按以下顺序落地：

1. 建立 `roll/server/` 基础包结构
2. 完成 FastAPI 应用、数据库连接、JWT、RBAC
3. 完成 `Project / Template / Job / Run` 基础模型
4. 实现 `LocalProcessExecutor`
5. 打通 RLVR 和 Agentic 两类 pipeline 的最小提交流程
6. 打通日志流、状态流、配置快照和 artifact 查询
7. 补充 Prometheus 指标和审计
8. 再扩展 Distill、DPO、SFT、RewardFL

这是最稳的路径，因为 RLVR 和 Agentic 最能代表当前 ROLL 的核心场景。

---

## 二十四、默认假设

- 该后端面向内部研发或训练平台，不是公网 SaaS
- v1 是单集群控制面，但数据库中保留 `Cluster` 抽象
- v1 采用基础 RBAC，不做完整多租户
- v1 不在 API 进程内直接运行 ROLL pipeline
- 主提交流程是模板化参数提交，不鼓励任意 YAML 原样透传
- checkpoint 与大文件产物仍以 ROLL 原生输出和对象存储为主，控制面维护索引和元数据

---

## 结论

结合 ROLL 当前的架构和功能，最合理的后端方案不是另起一套训练框架，而是构建一个基于 **FastAPI** 的统一控制面，围绕 `ROLL + Ray` 提供任务编排、配置治理、状态观测、权限控制和平台 API。

对于 v1，推荐采用：

- **FastAPI**
- **PostgreSQL**
- **Redis**
- **Celery**
- **Prometheus + Grafana**
- **MinIO / S3**

并以“单集群控制面 + 基础 RBAC + 模板化作业提交”为最小可落地形态，在此基础上逐步演进到多集群平台。
