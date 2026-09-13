# mini-SWE / Qwen3.6 scale metrics

| Metric | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Task throughput (tasks/min) | 0.23 | 0.43 | 0.80 | 1.30 | 1.90 | 2.30 | 0.50 |
| Completed in window | 7 | 13 | 24 | 39 | 57 | 69 | 15 |
| Admitted in window / latency n | 7 | 13 | 24 | 39 | 57 | 69 | 15 |
| Unique admitted traces | 7 | 13 | 24 | 39 | 52 | 48 | 15 |
| Task latency mean (s) | 256.98 | 299.77 | 319.18 | 351.09 | 430.69 | 747.57 | 4054.15 |
| Task latency p50 (s) | 225.18 | 308.94 | 269.94 | 292.47 | 379.30 | 638.56 | 4068.61 |
| Task latency p95 (s) | 404.28 | 479.36 | 642.48 | 766.32 | 910.05 | 1518.09 | 4719.72 |
| Output throughput: full window (tokens/s) | 74.70 | 153.25 | 246.81 | 370.44 | 563.83 | 715.87 | 342.44 |
| Output throughput: active backend (tokens/s) | 150.96 | 187.31 | 255.65 | 371.14 | 563.83 | 715.87 | 342.44 |
| Decode throughput: active decode (tokens/s) | 166.77 | 196.13 | 258.55 | 369.54 | 560.82 | 712.01 | 340.59 |
| LLM calls started/s | 0.389 | 0.688 | 1.276 | 2.067 | 3.014 | 3.857 | 1.878 |
| Tool calls started/s | 0.389 | 0.688 | 1.274 | 2.067 | 3.014 | 3.858 | 1.873 |
| LLM latency mean (s) | 1.329 | 1.795 | 1.989 | 2.570 | 4.025 | 7.067 | 33.635 |
| LLM latency p95 (s) | 5.007 | 6.809 | 6.661 | 8.771 | 13.852 | 24.724 | 87.266 |
| Client TTFT p95 (s) | 0.306 | 0.334 | 0.371 | 0.386 | 0.428 | 1.142 | 57.706 |
| Client TPOT estimate mean (ms/token) | 4.66 | 5.66 | 7.31 | 10.31 | 15.49 | 27.53 | 54.34 |
| Initial backend queue p95 (s) | 0.00002 | 0.00002 | 0.00001 | 0.00001 | 0.00001 | 0.21741 | 55.15059 |
| Scheduled to first token p95 (s) | 0.176 | 0.206 | 0.225 | 0.251 | 0.261 | 0.343 | 1.954 |
| Tool latency p95 (s) | 1.467 | 1.326 | 1.496 | 1.733 | 1.756 | 1.710 | 1.815 |
| GPU busy: four-card mean (%) | 47.66 | 80.23 | 93.42 | 96.20 | 95.92 | 96.02 | 96.57 |
| GPU memory busy: four-card mean (%) | 12.35 | 22.03 | 29.99 | 38.41 | 46.71 | 51.33 | 45.76 |
| Allocated memory per GPU mean (GiB) | 37.67 | 37.67 | 37.67 | 37.67 | 37.67 | 37.67 | 37.71 |
| Four-GPU power sum (W) | 467.1 | 630.8 | 734.8 | 878.6 | 993.6 | 1100.0 | 1214.2 |
| Backend active time (%) | 49.48 | 81.82 | 96.54 | 99.81 | 100.00 | 100.00 | 100.00 |
| Backend running requests mean | 0.46 | 1.16 | 2.41 | 5.10 | 11.80 | 26.28 | 25.75 |
| Backend waiting requests mean | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.35 | 35.94 |
| Backend waiting requests max | 0 | 0 | 0 | 1 | 1 | 13 | 55 |
| KV occupancy mean (%) | 0.82 | 2.13 | 4.30 | 8.63 | 19.53 | 40.53 | 27.29 |
| KV occupancy max (%) | 4.08 | 7.77 | 11.55 | 17.25 | 31.63 | 60.32 | 63.46 |
| Prefix lookup hit ratio (%) | 97.24 | 97.39 | 97.51 | 97.52 | 97.35 | 95.05 | 17.77 |
| Prompt tokens actually reused (%) | 97.24 | 97.39 | 97.51 | 97.52 | 97.36 | 95.76 | 45.58 |
| Lookup tokens / processed prompt tokens | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.24 | 8.40 |
| Prefill compute tokens / sampled second | 302.09 | 552.34 | 972.22 | 1501.12 | 2262.72 | 4434.25 | 17475.34 |
| Preemptions (sample delta) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Fully cached last-token recomputes (sample delta) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Replay-node CPU mean (%) | 0.95 | 1.44 | 3.82 | 6.43 | 9.32 | 11.57 | 4.94 |
| Inference-node CPU mean (%) | 6.45 | 8.85 | 9.46 | 9.50 | 9.44 | 9.33 | 8.33 |
| Native tool errors | 88 | 161 | 246 | 359 | 593 | 756 | 448 |
| New tool errors | 2 | 2 | 2 | 14 | 17 | 22 | 5 |
| New tool error share (%) | 0.285 | 0.161 | 0.087 | 0.376 | 0.313 | 0.317 | 0.148 |
| Cohort tasks finishing after window | 1 | 2 | 4 | 8 | 16 | 32 | 15 |
| Warmup tasks completing in window | 1 | 2 | 4 | 8 | 16 | 32 | 15 |
| Window admissions completing in window | 6 | 11 | 20 | 31 | 41 | 37 | 0 |
| Setup mean (s) | 25.49 | 25.07 | 26.17 | 25.98 | 26.82 | 26.71 | 27.23 |
| Setup p95 (s) | 31.48 | 29.12 | 36.46 | 36.36 | 37.50 | 38.67 | 37.36 |
| LLM wall-time mean per admitted task (s) | 132.79 | 181.77 | 196.72 | 232.54 | 314.94 | 627.37 | 3923.13 |
| Tool wall-time mean per admitted task (s) | 96.03 | 90.32 | 93.65 | 90.02 | 86.33 | 90.93 | 101.12 |
| Mean LLM calls per admitted trace | 100.29 | 106.38 | 101.67 | 92.31 | 86.42 | 93.61 | 107.47 |
| Mean output tokens per admitted trace | 19137.4 | 22624.5 | 19246.2 | 16958.2 | 15278.5 | 17658.1 | 20271.7 |
| Sandbox build error log count | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Natural drain (s) | 16.5 | 318.6 | 452.4 | 541.4 | 595.8 | 997.5 | 3548.4 |
| Backend startup (s) | 236.9 | 102.4 | 102.4 | 106.6 | 231.2 | 97.4 | 100.4 |
