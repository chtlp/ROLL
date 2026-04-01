# vLLM Dataclass Bug 修复方案

## 问题描述

导入 vLLM 时，可能会遇到以下错误：

```
TypeError: non-default argument 'vision_config' follows default argument
```

此错误是因为某些 vLLM 配置类（如 `DeepseekVLV2Config`、`UltravoxConfig`）使用了 Python 的 `@dataclass` 装饰器，但类级别的类型注解没有默认值，却出现在有默认值的字段之前，导致生成的 `__init__` 方法参数顺序不符合 Python 规范。

### 受影响的配置类

| 配置类 | 有问题的字段 |
|--------|-------------|
| `DeepseekVLV2Config` | `vision_config`, `projector_config` |
| `UltravoxConfig` | `wrapped_model_config` |

## 根本原因

在 vLLM 源代码中，这些类定义了如下类型注解：

```python
class DeepseekVLV2Config(PretrainedConfig):
    vision_config: VisionEncoderConfig          # 无默认值
    projector_config: MlpProjectorConfig        # 无默认值

    tile_tag: str = "2D"                        # 有默认值
```

`@dataclass` 装饰器按顺序处理注解，将所有无默认值的字段放在有默认值字段之前，这违反了 Python 函数参数的规则。

## 解决方案

修复方法是从 vLLM site-packages 安装中的受影响文件中移除不必要的类级别类型注解：

### 文件 1: `vllm/transformers_utils/configs/deepseek_vl2.py`

```python
class DeepseekVLV2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"
    # 已移除: vision_config: VisionEncoderConfig
    # 已移除: projector_config: MlpProjectorConfig

    tile_tag: str = "2D"
    # ...
```

### 文件 2: `vllm/transformers_utils/configs/ultravox.py`

```python
class UltravoxConfig(transformers.PretrainedConfig):
    model_type = "ultravox"
    # 已移除: wrapped_model_config: transformers.PretrainedConfig
    # ...
```

这些属性都在相应的 `__init__` 方法中正确初始化，因此类级别的注解是冗余的，且会导致 dataclass 问题。

## 手动修复命令

如果需要手动应用此修复：

```bash
# 修复 deepseek_vl2.py
sed -i '/^    vision_config: VisionEncoderConfig$/d' \
    .venv/lib/python3.12/site-packages/vllm/transformers_utils/configs/deepseek_vl2.py
sed -i '/^    projector_config: MlpProjectorConfig$/d' \
    .venv/lib/python3.12/site-packages/vllm/transformers_utils/configs/deepseek_vl2.py

# 修复 ultravox.py
sed -i '/^    wrapped_model_config: transformers.PretrainedConfig$/d' \
    .venv/lib/python3.12/site-packages/vllm/transformers_utils/configs/ultravox.py
```

## 注意事项

这是 vLLM 库本身的 bug。如果重新安装或升级 vLLM，此修复将会失效。建议向 vLLM 维护者报告此问题。
