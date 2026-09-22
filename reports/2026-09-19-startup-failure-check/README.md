# 2026-09-19：n16 重交后的启动失败

本次三次提交均未创建 `control/start.json`，`workers/` 为空，没有进入 replay 或测量窗口。

| 作业 | PBS walltime | 结论 |
|---|---:|---|
| 7637711 / OpenClaw | 1 分 54 秒 | Exit 1；r08 在 CUDA 设备初始化时失败 |
| 7637713 / mini-SWE | 1 分 23 秒 | Exit 1；多个 TP worker 被 Hugging Face API 429 限流，同时 r08 再次出现同样的 CUDA 初始化失败 |
| 7637723 / mini-SWE | 13 秒 | PBS comment 记录为用户终止；SSH 255 和 CancelledError 是终止后的结果 |

PBS 证据见 [pbs-status.json](pbs-status.json)。这与此前在 replay 请求阶段遇到的 ReadError 发生在不同阶段；此次修复前的本机 HTTP 回归测试没有覆盖 GPU 初始化和真实容器的模型加载路径。

## GPU 设备初始化

两次失败的 r08 都是 `x3208c0s37b1n0`。在 `torch.accelerator.set_device_index` 处报：

```text
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

原始证据：

- `runs/scaling/multinode/openclaw-fixed_frontend-n16-7637711.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/backends/r08/service.log:79`
- `runs/scaling/multinode/minisweagent-fixed_frontend-n16-7637713.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/backends/r08/service.log:79`

这不是 KV cache 使用率的测量结果，也没有日志证据支持显存 OOM。现有日志不能进一步区分设备占用、残留上下文或驱动/设备异常。检查时 PBS 已将该节点分给其他作业；本次没有 SSH 到该节点，没有操作其进程或 GPU，也未证明设备已经恢复。

## 已修复本地缓存仍触发联网的问题

mini-SWE r00 的 traceback 显示，Transformers 在 tokenizer 的 `_patch_mistral_regex → is_base_mistral → model_info` 路径访问 `https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B`，得到 `429 Too Many Requests`。后续的 `Unable to load vocabulary from file` 是包装后的异常，不能据此判定 tokenizer 文件损坏。其他多个 replica 也出现相同限流。

原启动参数使用 Hub repo ID，即使权重已缓存，仍可能执行远程模型元数据查询。现在共享启动脚本先从已有 cache 的 `refs/main` 解析本地 snapshot，将该目录传给 vLLM，并在宿主及容器中设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。保留 `served-model-name`，没有改变模型权重、TP4 或测量参数。缓存缺失时直接说明缺少本地缓存，不在正式作业中尝试下载。

验证结果：

- 本地 26 个 safetensors 分片均存在且非空；配置和 tokenizer 文件存在。
- **实际 vLLM 0.19.1 容器内 CPU 验证通过**：阻止所有 socket connect 后，AutoConfig、AutoTokenizer 加载并编码成功。未加载模型权重或使用 GPU。结果见 [offline-tokenizer-validation.json](offline-tokenizer-validation.json)。
- `tests/test_vllm_launcher.py`、`tests/test_vllm_backend.py` 共 **15 项通过**；shell 语法检查通过。测试覆盖真实脚本的本地 snapshot 参数、容器离线环境变量、模型服务名，以及缓存缺失时不启动容器。

现有排队脚本会加载这一共享启动脚本，无需因离线修复重交。此次没有提交、取消作业或启动 GPU 实验。GPU 初始化问题仍待在有权使用该节点时进一步检查，不能把此修复描述成 n16 满载流程已经验证通过。
