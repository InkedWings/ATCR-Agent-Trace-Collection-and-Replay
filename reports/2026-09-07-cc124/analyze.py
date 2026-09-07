"""Reproduce the cc=1/2/4 review without modifying experiment inputs."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUN = REPO / 'runs/scaling/single-backend-cc124-20260907-2'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def csv_file(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


rows, task_rows, checks = [], [], []
traces = {}
for workload in ('openclaw', 'minisweagent'):
    for cc in (1, 2, 4):
        point = RUN / f'{workload}-cc{cc}'
        root = point / 'benchmark'
        data, manifest = read(root / 'summary.json'), read(root / 'manifest.json')
        m = data['measurement']
        start, end = m['started_unix'], m['ended_unix']
        assert data['status'] == 'completed' and m['valid']
        assert end - start == 600 and manifest['warmup_seconds'] == 120
        assert manifest['concurrency'] == cc and manifest['seed'] == 42
        assert read(point / 'service.json')['config'] == read(RUN / 'config.json')['serve']
        for path in manifest['trace_pool']:
            p = Path(path)
            assert p.is_file()
            if path not in traces:
                trace = read(p)
                assert trace['schema_version'] == 2 and trace['trace_id']
                nodes = {n['id']: n for n in trace['nodes']}
                assert len(nodes) == len(trace['nodes'])
                for n in nodes.values():
                    assert n['type'] in ('llm', 'tool')
                    assert all(parent in nodes for parent in n['depends_on'])
                seed = trace['context'].get('workspace_seed')
                if seed:
                    assert (p.parent / seed).is_dir()
                for artifact in trace['artifacts']:
                    assert (p.parent / artifact['path']).is_file()
                traces[path] = trace
        tool_all, tool_window = Counter(), Counter()
        client_tokens = llm_count = node_count = 0
        finished = sum(start <= t['ended_unix'] < end for t in data['tasks'])
        admitted = sum(start <= t['started_unix'] < end for t in data['tasks'])
        assert (finished, admitted) == (m['completed_in_window'], m['admitted_in_window'])
        assert math.isclose(finished / 600, m['task_throughput_per_second'])
        for task in data['tasks']:
            assert task['status'] == 'completed' and task['returncode'] == 0
            report = read(root / task['report'])
            source = traces[task['trace_path']]
            original = {n['id']: n for n in source['nodes']}
            measured = {n['node_id']: n for n in report['nodes']}
            assert len(measured) == len(report['nodes']) and measured.keys() == original.keys()
            events = list(lines(root / 'tasks' / task['task_id'] / 'events.jsonl'))
            completed = [e for e in events if e['event'] == 'node_completed']
            assert len(completed) == len(measured)
            assert not any(e['event'] in ('node_failed', 'node_cancelled', 'replay_failed') for e in events)
            for n in report['nodes']:
                assert n['status'] == 'completed' and n['elapsed_seconds'] >= 0
                assert math.isfinite(n['elapsed_seconds'])
                for dep in original[n['node_id']]['depends_on']:
                    parent = measured[dep]
                    assert n['started_unix'] + .005 >= parent['started_unix'] + parent['elapsed_seconds']
                node_count += 1
                if n['type'] == 'llm':
                    assert n['actual_output_tokens'] == original[n['node_id']]['output_tokens'] == n['target_output_tokens']
                    client_tokens += n['actual_output_tokens']
                    llm_count += 1
                else:
                    key = f"{original[n['node_id']]['recorded_result']['isError']}->{n['native_error']}"
                    tool_all[key] += 1
                    if start <= n['started_unix'] < end:
                        tool_window[key] += 1
            task_rows.append(dict(point=point.name, workload=workload, cc=cc,
                task_id=task['task_id'], trace_id=source['trace_id'], cycle=task['cycle'],
                admission_phase=task['admission_phase'], completed_in_window=start <= task['ended_unix'] < end,
                lifecycle_s=task['lifecycle_seconds'], setup_s=report['setup_seconds'],
                replay_s=report['replay_makespan_seconds'],
                other_lifecycle_s=task['lifecycle_seconds']-report['setup_seconds']-report['replay_makespan_seconds']))
        backend_total = backend_window = backend_finished_tokens = 0
        backend_finished = set()
        for event in lines(point / 'backend.jsonl'):
            if event['event'] == 'tokens':
                backend_total += event['output_tokens']
                if start <= event['timestamp_unix'] < end:
                    backend_window += event['output_tokens']
            elif event['event'] == 'request_finished':
                assert event['request_id'] not in backend_finished
                backend_finished.add(event['request_id'])
                backend_finished_tokens += event['output_tokens']
        b = m['backend_throughput']
        assert len(backend_finished) == llm_count
        assert client_tokens == backend_total == backend_finished_tokens == data['actual_output_tokens']
        assert client_tokens == data['validation']['vllm_generation_counter_delta']
        assert backend_window == b['output_tokens']
        sample_checks = {}
        for name, path in [('replay', root / 'metrics.jsonl'), ('inference', point / 'inference-metrics.jsonl')]:
            samples = [s for s in lines(path) if start <= s['timestamp_unix'] < end]
            times = [s['timestamp_unix'] for s in samples]
            assert all(y > x for x, y in zip(times, times[1:]))
            sample_checks[name] = dict(count=len(samples), max_gap_s=max(y-x for x,y in zip(times,times[1:])),
                start_gap_s=times[0]-start, end_gap_s=end-times[-1])
        gauges = m['metrics']['vllm_gauge_sample_statistics']
        kv = next(v for k,v in gauges.items() if k.startswith('vllm:kv_cache_usage_perc'))
        waiting = next(v for k,v in gauges.items() if k.startswith('vllm:num_requests_waiting'))
        hw = m['inference_metrics']['hardware_sample_statistics']
        rows.append(dict(point=point.name, workload=workload, cc=cc, valid=m['valid'],
            completed_in_window=finished, tasks_per_min=finished/10,
            lifecycle_mean_s=m['task_lifecycle_seconds']['mean'], lifecycle_p95_s=m['task_lifecycle_seconds']['p95'],
            lifecycle_n=m['task_lifecycle_seconds']['count'], admitted_unique_traces=len(m['admitted_trace_coverage']),
            pool_size=len(manifest['trace_pool']), setup_mean_s=m['task_setup_seconds']['mean'],
            window_output_tokens_s=backend_window/600, backend_active_output_tokens_s=b['active_output_tokens_per_second'],
            backend_active_percent=b['active_seconds']/6,
            gpu_busy_mean_percent=mean(hw[f'gpu.{i}.gpu_busy_percent']['mean'] for i in range(4)),
            gpu_power_sum_w=sum(hw[f'gpu.{i}.power_watts']['mean'] for i in range(4)),
            llm_concurrency_mean=m['concurrency']['llm_calls']['time_weighted_mean'],
            waiting_max=waiting['max'], kv_mean_percent=kv['mean']*100, kv_max_percent=kv['max']*100,
            prefix_hit_percent=m['metrics']['vllm']['prefix_cache_hit_ratio']*100,
            ttft_p95_s=m['llm_ttft_seconds']['p95'], tpot_mean_ms=m['llm_tpot_estimate_seconds']['mean']*1000,
            tool_calls=m['tool_latency_seconds']['count'], native_tool_errors=m['native_tool_errors'],
            native_tool_error_percent=m['native_tool_errors']/m['tool_latency_seconds']['count']*100,
            new_tool_errors=tool_window['False->True'], original_errors_now_success=tool_window['True->False']))
        checks.append(dict(point=point.name, tasks=len(data['tasks']), nodes=node_count, llm_calls=llm_count,
            client_and_backend_total_tokens=client_tokens, backend_window_tokens=backend_window,
            samples=sample_checks, tool_flags_all=dict(tool_all), tool_flags_window=dict(tool_window)))

csv_file('summary.csv', rows)
csv_file('tasks.csv', task_rows)
(OUT / 'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
print(json.dumps(checks, indent=2))

# Paper-style diagnostic plots, with sample-size and measurement caveats explicit.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
panels = [('tasks_per_min','Completed tasks / min'), ('window_output_tokens_s','Output tokens / wall-clock s'),
          ('gpu_busy_mean_percent','Mean inference GPU busy (%)'), ('prefix_hit_percent','Prefix-cache hit (%)'),
          ('lifecycle_p95_s','Task lifecycle p95 (s; small samples)'), ('setup_mean_s','Mean task setup (s)')]
for ax, (key, label) in zip(axes.flat, panels):
    for workload, color, title in [('openclaw','#2271B2','OpenClaw / GAIA'), ('minisweagent','#D55E00','mini-SWE / SWE-bench')]:
        selected = [r for r in rows if r['workload']==workload]
        ax.plot([r['cc'] for r in selected], [r[key] for r in selected], '-o', color=color, label=title)
        if key=='lifecycle_p95_s':
            for r in selected:
                ax.annotate(f"n={r['lifecycle_n']}", (r['cc'],r[key]),
                            xytext=(4,-13 if workload=='openclaw' else 6),
                            textcoords='offset points', fontsize=8, color=color)
    ax.set(xlabel='Concurrent task lifecycles', ylabel=label, xticks=[1,2,4])
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('Single inference node: cc=1, 2, 4\nQwen3-32B BF16, TP=4; 120 s warmup + 600 s measurement; one run / point', fontsize=13)
fig.text(.5,.01,'Exploratory results: workload mix and cache warmth vary across windows; p95 is not a capacity/SLO estimate.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.93))
fig.savefig(OUT / 'overview.png',dpi=170)
fig.savefig(OUT / 'overview.pdf')
plt.close(fig)
