"""Aggregate the ten valid scale points; preserve missing cc=16 as missing."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ROOT = REPO.parent
BASE = REPO / 'runs/scaling'
LOW = BASE / 'single-backend-cc124-20260907-2'


def read(path):
    return json.loads(path.read_text())


def run_path(workload, cc):
    if cc < 8:
        return LOW
    if cc == 32:
        return BASE / 'preemptable-cc32-7596641/experiment'
    if workload == 'openclaw':
        return BASE / 'preemptable-cc8-7596639/experiment'
    return BASE / 'debug-cc8-7597420/experiment'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


rows, checks, task_rows = [], [], []
trace_cache = {}
serve = read(LOW / 'config.json')['serve']
for workload in ('openclaw', 'minisweagent'):
    baseline_manifest = read(LOW / f'{workload}-cc1/benchmark/manifest.json')
    for cc in (1, 2, 4, 8, 32):
        run = run_path(workload, cc)
        point = run / f'{workload}-cc{cc}'
        root = point / 'benchmark'
        d, manifest, config = read(root / 'summary.json'), read(root / 'manifest.json'), read(run / 'config.json')
        m = d['measurement']
        assert d['status'] == 'completed' and m['valid'] and not d['dry_run']
        assert config['serve'] == serve == read(point / 'service.json')['config']
        for key in ('trace_pool', 'seed', 'warmup_seconds', 'duration_seconds', 'llm_replay', 'concurrency_scope'):
            assert manifest[key] == baseline_manifest[key], (point.name, key)
        assert manifest['concurrency'] == cc
        duration = m['ended_unix'] - m['started_unix']
        assert duration == 600 and m['incomplete_admission_cohort'] == 0
        for name in ('metrics', 'inference_metrics'):
            assert m[name]['error_samples'] == 0
            assert m[name]['gpu_status_counts'] == {'ok': m[name]['samples']}
        for name in ('backend_client_counts_match', 'counter_counts_match', 'inference_gpu_samples_ok', 'window_has_backend_tokens'):
            assert d['validation'][name]
        assert all(Path(p).is_file() for p in manifest['trace_pool'])
        finished = sum(m['started_unix'] <= t['ended_unix'] < m['ended_unix'] for t in d['tasks'])
        admitted = sum(m['started_unix'] <= t['started_unix'] < m['ended_unix'] for t in d['tasks'])
        assert finished == m['completed_in_window'] and admitted == m['admitted_in_window']
        assert math.isclose(finished / duration, m['task_throughput_per_second'])
        all_llm = all_tokens = all_nodes = 0
        window_llm = window_tool = 0
        errors_all, errors_window = Counter(), Counter()
        for t in d['tasks']:
            assert t['status'] == 'completed' and t['returncode'] == 0
            report = read(root / t['report'])
            if t['trace_path'] not in trace_cache:
                trace_cache[t['trace_path']] = read(Path(t['trace_path']))
            trace = trace_cache[t['trace_path']]
            original = {n['id']: n for n in trace['nodes']}
            assert len(report['nodes']) == len(original)
            assert {n['node_id'] for n in report['nodes']} == set(original)
            for n in report['nodes']:
                all_nodes += 1
                assert n['status'] == 'completed'
                in_window = m['started_unix'] <= n['started_unix'] < m['ended_unix']
                if n['type'] == 'llm':
                    all_llm += 1
                    all_tokens += n['actual_output_tokens']
                    assert n['actual_output_tokens'] == n['target_output_tokens'] == original[n['node_id']]['output_tokens']
                    window_llm += in_window
                else:
                    window_tool += in_window
                    key = f"{original[n['node_id']]['recorded_result']['isError']}->{n['native_error']}"
                    errors_all[key] += 1
                    if in_window:
                        errors_window[key] += 1
            task_rows.append(dict(workload=workload, cc=cc, task_id=t['task_id'], trace_id=trace['trace_id'],
                cycle=t['cycle'], admission_phase=t['admission_phase'],
                lifecycle_s=t['lifecycle_seconds'], setup_s=report['setup_seconds'],
                replay_s=report['replay_makespan_seconds'],
                other_lifecycle_s=t['lifecycle_seconds']-report['setup_seconds']-report['replay_makespan_seconds']))
        assert all_tokens == d['actual_output_tokens'] == d['target_output_tokens'] == d['validation']['vllm_generation_counter_delta']
        assert all_llm == d['llm_latency_seconds']['count']
        assert window_llm == m['llm_latency_seconds']['count']
        assert window_tool == m['tool_latency_seconds']['count']
        assert errors_window['False->True'] + errors_window['True->True'] == m['native_tool_errors']
        b = m['backend_throughput']
        gauges = m['metrics']['vllm_gauge_sample_statistics']
        def gauge(prefix):
            return next(value for key,value in gauges.items() if key.startswith(prefix+'{'))
        kv, waiting, running = (gauge(x) for x in ('vllm:kv_cache_usage_perc','vllm:num_requests_waiting','vllm:num_requests_running'))
        hw, replay_hw = m['inference_metrics']['hardware_sample_statistics'], m['metrics']['hardware_sample_statistics']
        counters = m['metrics']['vllm']['counter_deltas']
        sample_span = m['metrics']['vllm']['last_sample_unix'] - m['metrics']['vllm']['first_sample_unix']
        prompt_delta = sum(v for k,v in counters.items() if k.startswith('vllm:prompt_tokens_total{'))
        row = dict(workload=workload, cc=cc, valid=True, run=str(run.relative_to(ROOT)),
            inference_node=config['inference_node'], replay_node=config['replay_node'],
            completed_in_window=finished, admitted_in_window=admitted,
            task_latency_n=m['task_lifecycle_seconds']['count'], pool_size=len(manifest['trace_pool']),
            admitted_unique_traces=len(m['admitted_trace_coverage']), tasks_per_min=finished/10,
            task_latency_mean_s=m['task_lifecycle_seconds']['mean'], task_latency_p50_s=m['task_lifecycle_seconds']['p50'],
            task_latency_p95_s=m['task_lifecycle_seconds']['p95'],
            setup_mean_s=m['task_setup_seconds']['mean'], replay_mean_s=m['task_replay_seconds']['mean'],
            other_lifecycle_mean_s=m['task_lifecycle_seconds']['mean']-m['task_setup_seconds']['mean']-m['task_replay_seconds']['mean'],
            llm_calls_started_per_s=window_llm/duration, tool_calls_started_per_s=window_tool/duration,
            prompt_counter_tokens_per_sample_s=prompt_delta/sample_span,
            output_tokens_per_window_s=b['window_output_tokens_per_second'],
            output_tokens_per_active_s=b['active_output_tokens_per_second'], decode_tokens_per_active_s=b['decode_tokens_per_second'],
            backend_active_percent=b['active_seconds']/duration*100,
            llm_latency_mean_s=m['llm_latency_seconds']['mean'], llm_latency_p95_s=m['llm_latency_seconds']['p95'],
            ttft_p95_ms=m['llm_ttft_seconds']['p95']*1000, tpot_mean_ms=m['llm_tpot_estimate_seconds']['mean']*1000,
            backend_queue_mean_ms=b['queue_seconds']['mean']*1000, backend_queue_p95_ms=b['queue_seconds']['p95']*1000,
            backend_prefill_p95_ms=b['prefill_seconds']['p95']*1000, backend_decode_p95_s=b['decode_latency_seconds']['p95'],
            tool_latency_mean_s=m['tool_latency_seconds']['mean'], tool_latency_p95_s=m['tool_latency_seconds']['p95'],
            tool_errors=m['native_tool_errors'], tool_calls=window_tool, tool_error_percent=100*m['native_tool_errors']/window_tool,
            new_tool_errors=errors_window['False->True'],
            gpu_busy_mean_percent=mean(hw[f'gpu.{i}.gpu_busy_percent']['mean'] for i in range(4)),
            gpu_memory_busy_mean_percent=mean(hw[f'gpu.{i}.memory_busy_percent']['mean'] for i in range(4)),
            gpu_allocated_mean_gib=mean(hw[f'gpu.{i}.memory_used_mib']['mean'] for i in range(4))/1024,
            gpu_power_sum_w=sum(hw[f'gpu.{i}.power_watts']['mean'] for i in range(4)),
            inference_cpu_mean_percent=hw['cpu_busy_percent']['mean'], replay_cpu_mean_percent=replay_hw['cpu_busy_percent']['mean'],
            task_concurrency_mean=m['concurrency']['tasks']['time_weighted_mean'],
            llm_concurrency_mean=m['concurrency']['llm_calls']['time_weighted_mean'],
            backend_running_mean=running['mean'], backend_waiting_mean=waiting['mean'], backend_waiting_max=waiting['max'],
            kv_mean_percent=kv['mean']*100, kv_max_percent=kv['max']*100,
            prefix_hit_percent=m['metrics']['vllm']['prefix_cache_hit_ratio']*100,
            replay_metric_samples=m['metrics']['samples'], inference_metric_samples=m['inference_metrics']['samples'],
            drain_s=d['ended_unix']-m['ended_unix'])
        rows.append(row)
        checks.append(dict(point=point.name, tasks=len(d['tasks']), nodes=all_nodes, llm_calls=all_llm,
            output_tokens=all_tokens, window_llm_calls=window_llm, window_tool_calls=window_tool,
            tool_error_flags_all=dict(errors_all), tool_error_flags_window=dict(errors_window)))

write_csv(OUT/'summary.csv',rows)
write_csv(OUT/'tasks.csv',task_rows)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')

fields = [
 ('窗口完成任务数 / 延迟样本数','completed_in_window',0),
 ('窗口启动的不同 trace 数','admitted_unique_traces',0),
 ('任务吞吐，任务/min','tasks_per_min',2),
 ('任务延迟 mean，s','task_latency_mean_s',2), ('任务延迟 p50，s','task_latency_p50_s',2),
 ('任务延迟 p95，s','task_latency_p95_s',2),
 ('任务 setup mean，s','setup_mean_s',2), ('任务 replay mean，s','replay_mean_s',2),
 ('其他生命周期 mean，s','other_lifecycle_mean_s',2),
 ('LLM 调用开始率，calls/s','llm_calls_started_per_s',3),
 ('Tool 调用开始率，calls/s','tool_calls_started_per_s',3),
 ('请求输入 token/s（含缓存命中）','prompt_counter_tokens_per_sample_s',1),
 ('窗口输出 token/s','output_tokens_per_window_s',2),
 ('后端活跃输出 token/s','output_tokens_per_active_s',2),
 ('后端 decode token/s','decode_tokens_per_active_s',2),
 ('后端活跃时间，%','backend_active_percent',2),
 ('LLM 延迟 mean，s','llm_latency_mean_s',2), ('LLM 延迟 p95，s','llm_latency_p95_s',2),
 ('TTFT p95，ms','ttft_p95_ms',1), ('TPOT mean，ms/token','tpot_mean_ms',2),
 ('后端 queue mean，ms','backend_queue_mean_ms',4), ('后端 queue p95，ms','backend_queue_p95_ms',4),
 ('后端 prefill p95，ms','backend_prefill_p95_ms',1), ('后端 decode 延迟 p95，s','backend_decode_p95_s',2),
 ('Tool 延迟 mean，s','tool_latency_mean_s',3), ('Tool 延迟 p95，s','tool_latency_p95_s',3),
 ('Tool 错误数','tool_errors',0), ('Tool 调用数','tool_calls',0), ('Tool 错误率，%','tool_error_percent',2),
 ('原成功→回放失败的 tool 数','new_tool_errors',0),
 ('GPU busy 四卡均值，%','gpu_busy_mean_percent',2),
 ('GPU memory busy 四卡均值，%','gpu_memory_busy_mean_percent',2),
 ('每卡已分配显存均值，GiB','gpu_allocated_mean_gib',2),
 ('四卡功率合计，W','gpu_power_sum_w',1),
 ('推理节点 CPU mean，%','inference_cpu_mean_percent',2), ('回放节点 CPU mean，%','replay_cpu_mean_percent',2),
 ('客户端任务平均并发','task_concurrency_mean',3), ('客户端 LLM 平均并发','llm_concurrency_mean',3),
 ('后端 running 平均数','backend_running_mean',3), ('后端 waiting 平均数','backend_waiting_mean',3),
 ('后端 waiting 最大数','backend_waiting_max',0),
 ('KV 占用 mean，%','kv_mean_percent',2), ('KV 占用 max，%','kv_max_percent',2),
 ('Prefix-cache hit，%','prefix_hit_percent',2),
 ('回放节点采样数','replay_metric_samples',0), ('推理节点采样数','inference_metric_samples',0),
 ('自然 drain，s','drain_s',1)]
md=['# Scale 指标汇总：cc=1、2、4、8、32\n',
    '10 个有效点；cc=16 留空。120s warmup + 600s measurement；每点一次。\n',
    '窗口完成任务数与延迟样本数在这十个点数值恰好相同，但不一定是同一批任务。\n']
for workload in ('openclaw','minisweagent'):
    selected={r['cc']:r for r in rows if r['workload']==workload}
    md += [f'## {workload}\n','| 指标 | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,key,digits in fields:
        values=[f'{selected[cc][key]:.{digits}f}' if cc in selected else '—' for cc in (1,2,4,8,16,32)]
        md.append('| '+label+' | '+' | '.join(values)+' |')
    md.append('')
(OUT/'tables.md').write_text('\n'.join(md).rstrip()+'\n')
print('All ten selected points passed configuration, task/report counts and saved validation checks.')
print('Tasks:',sum(c['tasks'] for c in checks),'nodes:',sum(c['nodes'] for c in checks),'tokens:',sum(c['output_tokens'] for c in checks))
print('\n'.join(md))
