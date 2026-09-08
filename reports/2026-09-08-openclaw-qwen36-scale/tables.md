# OpenClaw / Qwen3.6 scale metrics

| Metric | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Task throughput (tasks/min) | 1.33 | 2.30 | 3.67 | 4.90 | 5.97 | 4.67 | 3.43 |
| Completed in window | 40 | 69 | 110 | 147 | 179 | 140 | 103 |
| Admitted in window / latency n | 40 | 69 | 110 | 147 | 179 | 140 | 103 |
| Unique admitted traces | 40 | 69 | 101 | 101 | 107 | 100 | 81 |
| Task latency mean (s) | 45.53 | 52.57 | 65.77 | 97.92 | 160.36 | 431.15 | 1389.86 |
| Task latency p50 (s) | 35.10 | 39.05 | 43.54 | 58.13 | 94.79 | 239.78 | 1002.23 |
| Task latency p95 (s) | 113.36 | 116.57 | 190.77 | 301.59 | 505.58 | 1289.91 | 3613.81 |
| Output throughput: full window (tokens/s) | 72.94 | 131.13 | 232.53 | 330.18 | 410.95 | 289.24 | 184.88 |
| Output throughput: active backend (tokens/s) | 144.66 | 165.18 | 238.81 | 330.55 | 410.95 | 289.24 | 184.88 |
| Decode throughput: active decode (tokens/s) | 166.56 | 179.60 | 241.93 | 329.57 | 409.48 | 288.15 | 184.26 |
| LLM calls started/s | 0.301 | 0.576 | 0.858 | 1.186 | 1.468 | 1.083 | 0.644 |
| Tool calls started/s | 0.284 | 0.541 | 0.799 | 1.107 | 1.377 | 1.011 | 0.588 |
| LLM latency mean (s) | 1.741 | 1.990 | 3.132 | 5.201 | 9.440 | 28.665 | 100.717 |
| LLM latency p95 (s) | 4.679 | 5.672 | 9.410 | 17.218 | 31.644 | 98.844 | 191.032 |
| Client TTFT p95 (s) | 0.555 | 0.636 | 0.844 | 1.119 | 1.652 | 38.367 | 106.671 |
| Client TPOT estimate mean (ms/token) | 5.55 | 6.72 | 9.44 | 15.39 | 28.28 | 74.01 | 98.17 |
| Initial backend queue p95 (s) | 0.00002 | 0.00002 | 0.00001 | 0.11102 | 0.45364 | 35.53196 | 104.08484 |
| Scheduled to first token p95 (s) | 0.485 | 0.503 | 0.641 | 0.804 | 0.997 | 4.938 | 5.203 |
| Tool latency p95 (s) | 0.938 | 0.921 | 0.861 | 0.815 | 0.863 | 0.865 | 0.861 |
| GPU busy: four-card mean (%) | 48.49 | 77.95 | 95.83 | 98.42 | 98.66 | 98.49 | 97.44 |
| GPU memory busy: four-card mean (%) | 12.92 | 22.29 | 32.28 | 39.65 | 41.87 | 38.33 | 37.02 |
| Allocated memory per GPU mean (GiB) | 37.67 | 37.67 | 37.67 | 37.67 | 37.67 | 37.67 | 37.70 |
| Four-GPU power sum (W) | 474.2 | 658.2 | 821.3 | 952.3 | 1014.1 | 1081.2 | 1172.6 |
| Backend active time (%) | 50.42 | 79.39 | 97.37 | 99.89 | 100.00 | 100.00 | 100.00 |
| Backend running requests mean | 0.47 | 1.08 | 2.58 | 5.99 | 13.37 | 25.25 | 20.47 |
| Backend waiting requests mean | 0.00 | 0.00 | 0.01 | 0.04 | 0.11 | 4.75 | 42.18 |
| Backend waiting requests max | 0 | 1 | 2 | 3 | 4 | 20 | 53 |
| KV occupancy mean (%) | 0.91 | 2.75 | 6.18 | 14.55 | 30.24 | 50.34 | 28.22 |
| KV occupancy max (%) | 6.23 | 12.57 | 18.41 | 28.96 | 45.25 | 74.48 | 61.75 |
| Prefix lookup hit ratio (%) | 92.80 | 93.61 | 93.03 | 93.53 | 93.31 | 61.48 | 19.27 |
| Prompt tokens actually reused (%) | 92.80 | 93.71 | 93.46 | 94.11 | 93.77 | 81.14 | 21.79 |
| Lookup tokens / processed prompt tokens | 1.00 | 1.02 | 1.10 | 1.13 | 1.30 | 4.45 | 16.48 |
| Prefill compute tokens / sampled second | 793.79 | 1553.42 | 2433.49 | 3279.73 | 4253.58 | 8371.19 | 14971.23 |
| Preemptions (sample delta) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Fully cached last-token recomputes (sample delta) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Replay-node CPU mean (%) | 0.76 | 1.19 | 1.88 | 2.55 | 3.11 | 2.53 | 2.00 |
| Inference-node CPU mean (%) | 5.99 | 8.43 | 9.35 | 9.42 | 9.40 | 9.04 | 8.50 |
| Native tool errors | 41 | 75 | 107 | 154 | 189 | 148 | 85 |
| New tool errors | 1 | 2 | 4 | 7 | 4 | 2 | 2 |
| New tool error share (%) | 0.196 | 0.206 | 0.278 | 0.351 | 0.161 | 0.110 | 0.189 |
| Cohort tasks finishing after window | 1 | 2 | 4 | 8 | 16 | 32 | 51 |
| Natural drain (s) | 42.6 | 84.4 | 234.9 | 157.5 | 423.1 | 757.8 | 2888.0 |
| Backend startup (s) | 339.1 | 217.9 | 101.4 | 102.4 | 102.5 | 105.4 | 108.5 |
