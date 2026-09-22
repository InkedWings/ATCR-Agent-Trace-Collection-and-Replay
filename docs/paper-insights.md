# Paper Insights

## Insight 1: State retention can limit concurrency scaling

Agentic workloads alternate between LLM calls and tool execution while repeatedly reusing growing histories, making efficient execution depend on inference state retained between calls. In hybrid architectures combining attention and recurrent layers, prefix reuse requires both attention KV and the corresponding recurrent state. In the evaluated vLLM implementation, both are managed through a shared block pool: historical state blocks become reclaimable when their reference counts reach zero, and subsequent allocations can evict their cached contents. Higher concurrency can intensify this competition, leaving attention KV available but the required recurrent state missing. The resulting loss of prefix reuse forces additional prefill and can reduce output throughput, making state retention across calls a key constraint on agent concurrency scaling.

## Insight 2: Agentic workloads can achieve nearly linear weak scaling

Agentic workloads can scale nearly linearly when frontend resources and independent inference replicas grow together, task concurrency per replica remains fixed, and tasks retain affinity to their backends. In our experiments, scaling from one to sixteen inference replicas with matching frontend resources achieves approximately 99–101% scaling efficiency in output token throughput across both workloads while maintaining prefix reuse. These results show that dependencies between agent steps and intervening tool execution need not prevent scaling across nodes: replicating a stable serving configuration preserves local cache and request supply conditions while increasing aggregate throughput.
