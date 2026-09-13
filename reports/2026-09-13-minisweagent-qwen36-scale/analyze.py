"""Analyze the seven completed mini-SWE points; never run inference/tools.

Reuse the saved measurement windows and read raw Prometheus samples only for
additional counters and time trends. No experiment inputs are changed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/agenttrace-matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUN = REPO / 'runs/scaling/qwen36-expanded-7594555-20260908'
CCS = [1, 2, 4, 8, 16, 32, 64]
COLORS = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#333333']


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def replay_partition(report):
    """Disjoint client wall-time intervals, preserving any LLM/tool overlap."""
    start = report['replay_started_unix']
    end = start + report['replay_makespan_seconds']
    events = defaultdict(lambda: [0, 0])
    events[start]; events[end]
    for node in report['nodes']:
        a = max(start, node['started_unix'])
        b = min(end, node['started_unix'] + node['elapsed_seconds'])
        if b > a:
            k = 0 if node['type'] == 'llm' else 1
            events[a][k] += 1
            events[b][k] -= 1
    totals = {'llm_only_s': 0., 'tool_only_s': 0., 'overlap_s': 0., 'replay_gaps_s': 0.}
    active = [0, 0]
    previous = start
    for stamp, change in sorted(events.items()):
        key = ('overlap_s' if all(active) else 'llm_only_s' if active[0]
               else 'tool_only_s' if active[1] else 'replay_gaps_s')
        totals[key] += stamp-previous
        active = [a+b for a,b in zip(active, change)]
        previous = stamp
    assert math.isclose(sum(totals.values()), report['replay_makespan_seconds'], abs_tol=.001)
    return totals


PROM_NAMES = ['num_preemptions_total', 'prompt_tokens_recomputed_total', 'prompt_tokens_total',
              'prompt_tokens_cached_total',
              'generation_tokens_total', 'prefix_cache_hits_total', 'prefix_cache_queries_total',
              'num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc']
PROM_RE = re.compile(r'^vllm:('+'|'.join(PROM_NAMES)+r')(?:\{[^\n]*\})?\s+([^\s]+)', re.M)


def raw_metrics(path, start, end, cc, tasks):
    samples = []
    with path.open() as handle:
        for line in handle:
            sample = json.loads(line)
            stamp = sample['timestamp_unix']
            if stamp >= end:
                break
            if stamp < start:
                continue
            found = dict(PROM_RE.findall(sample['vllm_prometheus']))
            assert set(found) == set(PROM_NAMES)
            source_counts = {}
            for labels, value in re.findall(r'^vllm:prompt_tokens_by_source_total\{([^\n]*)\}\s+([^\s]+)', sample['vllm_prometheus'], re.M):
                source = re.search(r'source="([^"]+)"', labels).group(1)
                assert source not in source_counts
                source_counts[source] = float(value)
            assert set(source_counts) == {'local_compute', 'local_cache_hit', 'external_kv_transfer'}
            samples.append({'time': stamp, **{k: float(v) for k,v in found.items()},
                            **{'source_'+k: v for k,v in source_counts.items()}})
    assert len(samples) > 1700
    first, last = samples[0], samples[-1]
    counters = {k: last[k]-first[k] for k in PROM_NAMES if k.endswith('_total')}
    assert all(v >= 0 for v in counters.values())
    for source in ['local_compute','local_cache_hit','external_kv_transfer']:
        counters['source_'+source] = last['source_'+source]-first['source_'+source]
    assert counters['source_external_kv_transfer'] == 0
    assert counters['source_local_cache_hit'] == counters['prompt_tokens_cached_total']
    assert counters['source_local_compute'] + counters['source_local_cache_hit'] == counters['prompt_tokens_total']
    bins = []
    for i in range(30):
        left, right = start+i*60, start+(i+1)*60
        selected = [s for s in samples if left <= s['time'] < right]
        a,b = selected[0], selected[-1]
        dt = b['time']-a['time']
        delta = lambda key: b[key]-a[key]
        queries = delta('prefix_cache_queries_total')
        prompts = delta('prompt_tokens_total')
        cached = delta('prompt_tokens_cached_total')
        bins.append(dict(cc=cc, minute_start=i, minute_end=i+1, samples=len(selected),
            sampled_span_s=dt,
            output_tokens_per_s=delta('generation_tokens_total')/dt,
            prompt_tokens_per_s=delta('prompt_tokens_total')/dt,
            prefix_hit_percent=100*delta('prefix_cache_hits_total')/queries if queries else None,
            prompt_reuse_percent=100*cached/prompts if prompts else None,
            prefix_query_amplification=queries/prompts if prompts else None,
            prefill_compute_tokens_per_s=(prompts-cached)/dt,
            preemptions=delta('num_preemptions_total'),
            recomputed_tokens=delta('prompt_tokens_recomputed_total'),
            backend_running_mean=float(np.mean([s['num_requests_running'] for s in selected])),
            backend_waiting_mean=float(np.mean([s['num_requests_waiting'] for s in selected])),
            kv_mean_percent=100*float(np.mean([s['kv_cache_usage_perc'] for s in selected])),
            completed_tasks=sum(left <= t['ended_unix'] < right for t in tasks)))
    return counters, bins, last['time']-first['time']


def point_path(cc):
    branch = 'experiment' if cc < 16 else 'recovery-minisweagent-20260908T1330Z/experiment'
    return RUN / branch / f'minisweagent-cc{cc}'


def extract():
    summaries, task_rows, errors, checks, series = [], [], [], [], []
    config = read(RUN / 'config.json')
    baseline = read(RUN / 'experiment/minisweagent-cc1/benchmark/manifest.json')
    profile = read(Path(baseline['profile_path']))
    assert profile['tool_executor']['config'] == {'timeout': 60}
    config['analysis_provenance'] = dict(
        points={str(cc): str(point_path(cc).relative_to(REPO)) for cc in CCS},
        excluded_failed_point=str((RUN/'experiment/minisweagent-cc16').relative_to(REPO)),
        recovery_profile_change='sandbox_build_retries: 1 -> 3; build/setup only',
        trace_pool_size=61)
    traces = {}
    for cc in CCS:
        point = point_path(cc)
        root = point / 'benchmark'
        s, manifest = read(root/'summary.json'), read(root/'manifest.json')
        m, b = s['measurement'], s['measurement']['backend_throughput']
        start,end = m['started_unix'], m['ended_unix']
        assert s['status'] == 'completed' and m['valid'] and not s['dry_run']
        assert end-start == 1800 and manifest['concurrency'] == cc
        assert read(point/'service.json')['config'] == config['serve']
        for key in ['trace_pool','seed','warmup_seconds','duration_seconds','llm_replay','concurrency_scope']:
            assert manifest[key] == baseline[key], (cc,key)
        assert len(manifest['trace_pool']) == 61
        selected_profile = read(Path(manifest['profile_path']))
        attempts = selected_profile['tool_executor']['config'].pop('sandbox_build_retries', 1)
        assert attempts == (1 if cc < 16 else 3)
        assert selected_profile == profile
        assert m['incomplete_admission_cohort'] == 0 and b['status'] == 'measured'
        for key in ['backend_client_counts_match','counter_counts_match','inference_gpu_samples_ok','window_has_backend_tokens']:
            assert s['validation'][key]
        for key in ['metrics','inference_metrics']:
            assert m[key]['error_samples'] == 0
            assert m[key]['gpu_status_counts'] == {'ok':m[key]['samples']}
        total_tokens = total_llm = build_errors = 0
        window_llm = window_tools = 0
        transitions = Counter()
        for task in s['tasks']:
            assert task['status'] == 'completed' and task['returncode'] == 0
            path = task['trace_path']
            if path not in traces:
                trace = read(Path(path))
                traces[path] = {'id':trace['trace_id'], 'nodes':{n['id']:n for n in trace['nodes']}}
            trace = traces[path]
            original = trace['nodes']
            report = read(root/task['report'])
            console = (root/task['report']).with_name('console.log').read_text()
            build_errors += console.count('Error building image ')
            assert {n['node_id'] for n in report['nodes']} == set(original)
            admitted = start <= task['started_unix'] < end
            phases = replay_partition(report)
            cleanup = task['lifecycle_seconds']-report['setup_seconds']-report['replay_makespan_seconds']
            assert cleanup >= -.01
            task_rows.append(dict(cc=cc, task_id=task['task_id'], trace_id=trace['id'], cycle=task['cycle'],
                admission_phase=task['admission_phase'], in_admission_cohort=admitted,
                started_measurement_s=task['started_unix']-start, ended_measurement_s=task['ended_unix']-start,
                completed_in_window=start<=task['ended_unix']<end,
                finished_after_window=task['ended_unix']>=end,
                lifecycle_s=task['lifecycle_seconds'], setup_s=report['setup_seconds'],
                replay_s=report['replay_makespan_seconds'], cleanup_s=cleanup, **phases,
                llm_calls=sum(n['type']=='llm' for n in report['nodes']),
                output_tokens=sum(n.get('actual_output_tokens',0) for n in report['nodes'])))
            for node in report['nodes']:
                source = original[node['node_id']]
                assert node['status'] == 'completed'
                in_window = start <= node['started_unix'] < end
                if node['type'] == 'llm':
                    assert node['actual_output_tokens'] == node['target_output_tokens'] == source['output_tokens']
                    total_tokens += node['actual_output_tokens']; total_llm += 1
                    window_llm += in_window
                else:
                    assert source['request']['name'] == 'bash'
                    if in_window:
                        window_tools += 1
                        transitions[(source['request']['name'], bool(source['recorded_result']['isError']), bool(node['native_error']))] += 1
        for (name,before,after),count in sorted(transitions.items()):
            errors.append(dict(cc=cc, tool_name=name, capture_error=before, replay_error=after, calls=count))
        finished = sum(start<=t['ended_unix']<end for t in s['tasks'])
        admitted = sum(start<=t['started_unix']<end for t in s['tasks'])
        assert finished == m['completed_in_window'] and admitted == m['admitted_in_window'] == m['task_lifecycle_seconds']['count']
        assert total_tokens == s['actual_output_tokens'] == s['target_output_tokens'] == s['validation']['vllm_generation_counter_delta']
        assert total_llm == s['llm_latency_seconds']['count']
        assert window_llm == m['llm_latency_seconds']['count'] and window_tools == m['tool_latency_seconds']['count']
        assert sum(v for (name,before,after),v in transitions.items() if after) == m['native_tool_errors']
        assert math.isclose(finished/1800, m['task_throughput_per_second'])
        counters, bins, raw_span = raw_metrics(root/'metrics.jsonl', start, end, cc, s['tasks'])
        prompt_tokens = counters['prompt_tokens_total']
        cached_tokens = counters['prompt_tokens_cached_total']
        assert 0 <= cached_tokens <= prompt_tokens
        series.extend(bins)
        gauges = m['metrics']['vllm_gauge_sample_statistics']
        def gauge(name):
            return next(v for k,v in gauges.items() if k.split('{')[0] == 'vllm:'+name)
        hw = m['inference_metrics']['hardware_sample_statistics']
        cohort = [t for t in task_rows if t['cc']==cc and t['in_admission_cohort']]
        row = dict(cc=cc, valid=True, completed_tasks=finished, admitted_tasks=admitted,
            unique_admitted_traces=len(m['admitted_trace_coverage']),
            tasks_per_min=finished/30, task_mean_s=m['task_lifecycle_seconds']['mean'],
            task_p50_s=m['task_lifecycle_seconds']['p50'], task_p95_s=m['task_lifecycle_seconds']['p95'],
            cohort_finished_after_window=sum(t['finished_after_window'] for t in cohort),
            warmup_completed_in_window=sum(t['cc']==cc and not t['in_admission_cohort'] and t['completed_in_window'] for t in task_rows),
            cohort_completed_in_window=sum(t['completed_in_window'] for t in cohort),
            all_tasks=len(s['tasks']),
            build_error_log_count=build_errors,
            setup_p95_s=float(np.quantile([t['setup_s'] for t in cohort], .95)),
            mean_trace_llm_calls=float(np.mean([t['llm_calls'] for t in cohort])),
            mean_trace_output_tokens=float(np.mean([t['output_tokens'] for t in cohort])),
            llm_calls_started_per_s=window_llm/1800, tool_calls_started_per_s=window_tools/1800,
            output_tokens_per_window_s=b['window_output_tokens_per_second'],
            output_tokens_per_active_s=b['active_output_tokens_per_second'],
            decode_tokens_per_active_s=b['decode_tokens_per_second'],
            backend_active_percent=b['active_seconds']/1800*100,
            llm_mean_s=m['llm_latency_seconds']['mean'], llm_p95_s=m['llm_latency_seconds']['p95'],
            ttft_p95_s=m['llm_ttft_seconds']['p95'], tpot_mean_ms=m['llm_tpot_estimate_seconds']['mean']*1000,
            queue_mean_s=b['queue_seconds']['mean'], queue_p95_s=b['queue_seconds']['p95'],
            scheduled_to_first_token_p95_s=b['prefill_seconds']['p95'],
            decode_latency_p95_s=b['decode_latency_seconds']['p95'],
            tool_mean_s=m['tool_latency_seconds']['mean'], tool_p95_s=m['tool_latency_seconds']['p95'],
            native_tool_errors=m['native_tool_errors'], tool_calls=window_tools,
            new_tool_errors=sum(v for (name,before,after),v in transitions.items() if not before and after),
            gpu_busy_mean_percent=float(np.mean([hw[f'gpu.{i}.gpu_busy_percent']['mean'] for i in range(4)])),
            gpu_memory_busy_mean_percent=float(np.mean([hw[f'gpu.{i}.memory_busy_percent']['mean'] for i in range(4)])),
            gpu_allocated_mean_gib=float(np.mean([hw[f'gpu.{i}.memory_used_mib']['mean'] for i in range(4)]))/1024,
            gpu_power_sum_w=sum(hw[f'gpu.{i}.power_watts']['mean'] for i in range(4)),
            replay_cpu_mean_percent=m['metrics']['hardware_sample_statistics']['cpu_busy_percent']['mean'],
            inference_cpu_mean_percent=hw['cpu_busy_percent']['mean'],
            task_concurrency_mean=m['concurrency']['tasks']['time_weighted_mean'],
            client_llm_concurrency_mean=m['concurrency']['llm_calls']['time_weighted_mean'],
            backend_running_mean=gauge('num_requests_running')['mean'],
            backend_waiting_mean=gauge('num_requests_waiting')['mean'],
            backend_waiting_max=gauge('num_requests_waiting')['max'],
            kv_mean_percent=gauge('kv_cache_usage_perc')['mean']*100,
            kv_max_percent=gauge('kv_cache_usage_perc')['max']*100,
            prefix_hit_percent=m['metrics']['vllm']['prefix_cache_hit_ratio']*100,
            prompt_tokens_sample_delta=prompt_tokens,
            cached_prompt_tokens_sample_delta=cached_tokens,
            prefill_compute_tokens_sample_delta=prompt_tokens-cached_tokens,
            prompt_reuse_percent=100*cached_tokens/prompt_tokens,
            prefix_query_amplification=counters['prefix_cache_queries_total']/prompt_tokens,
            prefill_compute_tokens_per_sample_s=(prompt_tokens-cached_tokens)/raw_span,
            preemptions_sample_delta=counters['num_preemptions_total'],
            recomputed_tokens_sample_delta=counters['prompt_tokens_recomputed_total'],
            prompt_tokens_per_sample_s=counters['prompt_tokens_total']/raw_span,
            raw_counter_sample_span_s=raw_span,
            drain_s=s['ended_unix']-end,
            startup_s=read(point/'ready.json')['ready_unix']-read(point/'service.json')['started_unix'])
        for key in ['setup_s','llm_only_s','tool_only_s','overlap_s','replay_gaps_s','cleanup_s']:
            row[key.replace('_s','_mean_s')] = float(np.mean([t[key] for t in cohort]))
        assert np.allclose(np.quantile([t['lifecycle_s'] for t in cohort], [.5,.95]),
                           [row['task_p50_s'],row['task_p95_s']])
        assert math.isclose(np.mean([t['lifecycle_s'] for t in cohort]), row['task_mean_s'])
        summaries.append(row)
        checks.append(dict(cc=cc, completed_tasks_all=len(s['tasks']), llm_calls_all=total_llm,
            output_tokens_all=total_tokens, sandbox_build_attempts_allowed=attempts,
            build_error_log_count=build_errors,
            measurement_seconds=1800, cache_source_counters_match=True,
            source_counter_deltas={k:v for k,v in counters.items() if k.startswith('source_')},
            validation_passed=True))
        print(f"cc{cc}: {row['tasks_per_min']:.2f} tasks/min, preemptions={row['preemptions_sample_delta']:.0f}", flush=True)
    for r in summaries:
        r['speedup_vs_cc1'] = r['tasks_per_min']/summaries[0]['tasks_per_min']
        r['parallel_efficiency_percent'] = r['speedup_vs_cc1']/r['cc']*100
        r['new_tool_error_percent'] = r['new_tool_errors']/r['tool_calls']*100
    write_csv('summary.csv',summaries)
    write_csv('tasks.csv',task_rows)
    write_csv('tool_error_transitions.csv',errors)
    write_csv('timeseries_60s.csv',series)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    return summaries,task_rows,series,config


def paired_tasks(tasks):
    # Fixed launch prefix, including warmup; the measurement cohorts have no common trace.
    admitted = [{t['trace_id'] for t in tasks if t['cc']==cc and t['in_admission_cohort']} for cc in CCS]
    assert len(set.intersection(*admitted)) == 0
    groups = defaultdict(list)
    initial = {cc: sorted([t for t in tasks if t['cc']==cc], key=lambda t: t['task_id'])[:8] for cc in CCS}
    assert all([t['trace_id'] for t in initial[cc]] == [t['trace_id'] for t in initial[1]] for cc in CCS)
    for cc in CCS:
        for t in initial[cc]:
            groups[cc,t['trace_id']].append(t['lifecycle_s'])
    common = set.intersection(*[{tid for (c,tid) in groups if c==cc} for cc in CCS])
    result = []
    for trace_id in sorted(common):
        base = float(np.mean(groups[1,trace_id]))
        for cc in CCS:
            latency = float(np.mean(groups[cc,trace_id]))
            result.append(dict(trace_id=trace_id,cc=cc,n_replays=len(groups[cc,trace_id]),
                               admission_phase=next(t['admission_phase'] for t in initial[cc] if t['trace_id']==trace_id),
                               mean_lifecycle_s=latency,ratio_to_cc1=latency/base))
    assert result
    write_csv('initial_eight_paired_latency.csv',result)
    return result,len(common)


def audit_http_logs():
    """Count serving errors and distinguish unexpected probes from inference calls."""
    rows = []
    for cc in CCS:
        point = point_path(cc)
        summary = read(point/'benchmark/summary.json')
        files = list((point/'serve-logs').glob('*.log'))
        assert len(files) == 1
        lines = files[0].read_text().splitlines()
        chats = [line for line in lines if '"POST /v1/chat/completions ' in line]
        assert len(chats) == summary['llm_latency_seconds']['count']
        assert all('" 200 ' in line for line in chats)
        not_found = [i for i,line in enumerate(lines) if '" 404 ' in line]
        row = dict(cc=cc, chat_completions_http_200=len(chats),
                   http_404_all_run=len(not_found), first_404_previous_log_time_utc='',
                   first_404_next_log_time_utc='', measurement_end_utc=datetime.fromtimestamp(
                       summary['measurement']['ended_unix'], timezone.utc).isoformat())
        if not_found:
            def stamp(line):
                match = re.search(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d)',line)
                return match.group(1) if match else None
            row['first_404_previous_log_time_utc'] = next(stamp(l) for l in reversed(lines[:not_found[0]]) if stamp(l))
            row['first_404_next_log_time_utc'] = next(stamp(l) for l in lines[not_found[0]+1:] if stamp(l))
        rows.append(row)
    write_csv('http_log_checks.csv',rows)
    return rows


def plot(rows,tasks,series,paired,n_common):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'axes.titleweight':'bold'})
    x=np.arange(7)
    def cc_axis(ax,ylabel,title,log=False):
        ax.set(xticks=x,xticklabels=CCS,xlabel='Concurrent tasks (cc)',ylabel=ylabel,title=title)
        if log:ax.set_yscale('log')
        ax.grid(alpha=.18);ax.set_axisbelow(True)
    def line(ax,key,label,color='#4477AA',scale=1,ls='-'):
        ax.plot(x,[r[key]/scale for r in rows],marker='o',label=label,color=color,linestyle=ls,linewidth=2)
    figures=[]
    with PdfPages(OUT/'minisweagent_scale.pdf') as pdf:
        def save(fig,name,title,note,bottom=.065):
            fig.suptitle(title,fontsize=15,y=.99)
            fig.text(.5,.012,note,ha='center',va='bottom',fontsize=9)
            fig.tight_layout(rect=(0,bottom,1,.94),w_pad=2,h_pad=2)
            fig.savefig(OUT/f'{name}.png',dpi=180,facecolor='white')
            fig.savefig(OUT/f'{name}.pdf',facecolor='white')
            pdf.savefig(fig);plt.close(fig);figures.append(name)

        fig,ax=plt.subplots(2,3,figsize=(14,8))
        line(ax[0,0],'tasks_per_min','Task throughput');cc_axis(ax[0,0],'Tasks / minute','(a) Task throughput')
        for i,r in enumerate(rows):ax[0,0].annotate(f"{r['tasks_per_min']:.2f}",(i,r['tasks_per_min']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        ax[0,0].set_ylim(0,2.7)
        line(ax[0,1],'task_p50_s','Median',scale=60);line(ax[0,1],'task_p95_s','P95','#EE6677',scale=60)
        cc_axis(ax[0,1],'Task lifecycle (minutes)','(b) Task latency',True);ax[0,1].legend()
        line(ax[0,2],'output_tokens_per_window_s','Window output rate');cc_axis(ax[0,2],'Output tokens / second','(c) Inference output throughput')
        line(ax[1,0],'ttft_p95_s','Client TTFT');line(ax[1,0],'queue_p95_s','Backend initial queue','#EE6677')
        cc_axis(ax[1,0],'P95 seconds','(d) First-token and queue delay',True);ax[1,0].legend(fontsize=8)
        line(ax[1,1],'gpu_busy_mean_percent','GPU busy');line(ax[1,1],'backend_active_percent','Backend active','#228833')
        cc_axis(ax[1,1],'Window mean / time share (%)','(e) Inference utilization');ax[1,1].set_ylim(0,105);ax[1,1].legend(fontsize=8)
        line(ax[1,2],'prompt_reuse_percent','Prompt tokens reused')
        line(ax[1,2],'prefix_hit_percent','Lookup hit rate','#EE6677',ls='--')
        cc_axis(ax[1,2],'Token share (%)','(f) Cache reuse vs. cache lookups');ax[1,2].set_ylim(0,105);ax[1,2].legend(fontsize=8)
        save(fig,'01_scale_overview','mini-SWE scaling | Qwen3.6-35B-A3B, TP4, 61-trace pool',
             '120 s warmup + 1800 s measurement; one repetition per cc. Latencies follow window admissions through natural drain.\nNative bash tools; cc32 has the highest observed throughput. cc64 completions all started in warmup.')

        fig,ax=plt.subplots(1,2,figsize=(12,5.8))
        phases=['setup_mean_s','llm_only_mean_s','tool_only_mean_s','overlap_mean_s','replay_gaps_mean_s','cleanup_mean_s']
        labels=['Setup','LLM requests','Tool execution','LLM/tool overlap','Replay gaps','Cleanup / process overhead']
        palette=['#BBBBBB','#4477AA','#EE6677','#AA3377','#CCBB44','#66CCEE']
        mat=np.array([[r[k] for k in phases] for r in rows])
        assert np.allclose(mat.sum(axis=1),[r['task_mean_s'] for r in rows],atol=.01)
        for a,normalized in zip(ax,[False,True]):
            data=100*mat/mat.sum(axis=1)[:,None] if normalized else mat/60
            bottom=np.zeros(7)
            for j,(label,color) in enumerate(zip(labels,palette)):
                a.bar(x,data[:,j],bottom=bottom,color=color,label=label,width=.66);bottom+=data[:,j]
            cc_axis(a,'Share of mean lifecycle (%)' if normalized else 'Mean lifecycle (minutes)',
                    '(b) Lifecycle composition' if normalized else '(a) Mean lifecycle breakdown')
        handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.08),ncol=3,fontsize=8)
        save(fig,'02_latency_breakdown','Where task wall time is spent | disjoint client intervals',
             'LLM intervals include server waiting and network time. Parallel calls are counted by interval union; overlap is separate.',bottom=.19)

        fig,ax=plt.subplots(2,2,figsize=(12,8))
        line(ax[0,0],'backend_running_mean','Running');line(ax[0,0],'backend_waiting_mean','Waiting','#EE6677')
        cc_axis(ax[0,0],'Mean requests','(a) Scheduler occupancy');ax[0,0].legend()
        line(ax[0,1],'kv_mean_percent','Mean KV usage');line(ax[0,1],'kv_max_percent','Peak KV usage','#EE6677')
        cc_axis(ax[0,1],'Non-free KV blocks (%)','(b) Active KV-block occupancy');ax[0,1].set_ylim(0,100);ax[0,1].legend()
        ax[1,0].bar(x,[r['preemptions_sample_delta'] for r in rows],color='#AA3377')
        cc_axis(ax[1,0],'Preemption counter delta','(c) Engine preemptions')
        if all(r['preemptions_sample_delta']==0 for r in rows):
            ax[1,0].set(ylim=(0,1),yticks=[0,1])
            ax[1,0].plot(x,np.zeros(7),'o',color='#AA3377',clip_on=False)
            ax[1,0].text(.5,.55,'0 in every measured window',transform=ax[1,0].transAxes,ha='center',fontsize=12)
        line(ax[1,1],'tpot_mean_ms','Mean client TPOT estimate');cc_axis(ax[1,1],'Milliseconds / output token','(d) Output streaming slowdown')
        save(fig,'03_backend_pressure','Backend scheduling, cache and output streaming',
             'Free KV blocks can still hold evictable cached prefixes; this gauge does not measure all resident prefix data.\nCounters span ~1799 s. TPOT is estimated from client streaming chunks, not exact engine per-token timestamps.')

        fig,ax=plt.subplots(1,2,figsize=(12,6))
        cohorts=[[t['lifecycle_s']/60 for t in tasks if t['cc']==cc and t['in_admission_cohort']] for cc in CCS]
        bp=ax[0].boxplot(cohorts,positions=x,widths=.55,showfliers=False,patch_artist=True)
        rng=np.random.default_rng(42)
        for i,(v,box) in enumerate(zip(cohorts,bp['boxes'])):
            box.set(facecolor=COLORS[i],alpha=.35)
            ax[0].scatter(i+rng.uniform(-.15,.15,len(v)),v,s=9,color=COLORS[i],alpha=.45,edgecolors='none')
        cc_axis(ax[0],'Full task lifecycle (minutes)','(a) Admission-cohort distributions',True)
        ax[0].set_xticks(x,[f'{cc}\nn={len(v)}' for cc,v in zip(CCS,cohorts)])
        for tid in sorted({p['trace_id'] for p in paired}):
            ax[1].plot(x,[next(p['ratio_to_cc1'] for p in paired if p['cc']==cc and p['trace_id']==tid) for cc in CCS],color='#BBBBBB',alpha=.25,linewidth=.7)
        ratios=[[p['ratio_to_cc1'] for p in paired if p['cc']==cc] for cc in CCS]
        q=np.quantile(np.array(ratios),[.25,.5,.75],axis=1)
        ax[1].plot(x,q[1],color='#4477AA',marker='o',label='Median across shared traces')
        ax[1].fill_between(x,q[0],q[2],color='#4477AA',alpha=.15,label='IQR')
        cc_axis(ax[1],'Lifecycle / same-trace cc1',f'(b) First {n_common} launches (includes warmup)',True);ax[1].legend(fontsize=8)
        save(fig,'04_latency_distributions','Latency distributions and workload-mix sensitivity',
             'Boxes: median / IQR, whiskers 1.5 IQR; every observed task is shown. No censored task latency is dropped.\nRight: same first 8 tasks, one launch each; mixed warmup/measurement phases. Sensitivity check, not steady-state latency.')

        fig,ax=plt.subplots(2,2,figsize=(12,8))
        for cc,color in [(16,'#4477AA'),(32,'#EE6677'),(64,'#AA3377')]:
            selected=[s for s in series if s['cc']==cc]
            for a,key,label,title in [(ax[0,0],'output_tokens_per_s','Output tokens / second','(a) Output throughput'),
                (ax[0,1],'backend_waiting_mean','Mean waiting requests','(b) Queue accumulation'),
                (ax[1,0],'prompt_reuse_percent','Prompt tokens reused (%)','(c) Interval prompt-token reuse'),
                (ax[1,1],'completed_tasks','Tasks completed / minute','(d) Task completions')]:
                a.plot([s['minute_end'] for s in selected],[s[key] for s in selected],color=color,label=f'cc{cc}')
                a.set(xlabel='Minutes since measurement start',ylabel=label,title=title);a.grid(alpha=.18)
        for a in ax.flat:a.legend(fontsize=8)
        save(fig,'05_window_trends','Within-window trends | cc16, cc32 and cc64',
             '60-second bins; token/cache rates use sampled counter deltas within each bin (~59 s). Task counts use full 60 s bins.\nThese curves assess transient behavior; a 30-minute measurement does not by itself establish steady state.')

        fig,ax=plt.subplots(1,3,figsize=(14,5.5))
        line(ax[0],'tool_p95_s','Tool p95');cc_axis(ax[0],'Seconds','(a) Tool latency (p95)');ax[0].set_ylim(0,2.2)
        line(ax[1],'new_tool_error_percent','Capture success → replay error','#EE6677')
        cc_axis(ax[1],'Share of tool calls started in window (%)','(b) Newly failing tool calls');ax[1].set_ylim(bottom=0)
        line(ax[2],'unique_admitted_traces','Unique traces admitted');cc_axis(ax[2],'Unique traces out of 61','(c) Admission-cohort coverage');ax[2].set_ylim(0,65)
        save(fig,'06_tools_and_coverage','Tool behavior and sample coverage',
             'Native tool failures are retained; engine validation does not certify identical tool outcomes.\nAll points use the same trace pool and seed, but complete different task cohorts within a fixed window.')

        fig,ax=plt.subplots(1,3,figsize=(15,6))
        warm=np.array([r['warmup_completed_in_window'] for r in rows])
        inside=np.array([r['cohort_completed_in_window'] for r in rows])
        after=np.array([r['cohort_finished_after_window'] for r in rows])
        ax[0].bar(x,warm,color='#CCBB44',label='Started in warmup')
        ax[0].bar(x,inside,bottom=warm,color='#4477AA',label='Started in measurement')
        cc_axis(ax[0],'Tasks finished during 30-minute window','(a) Throughput sample origin')
        ax[0].legend(fontsize=8)
        ax[1].bar(x,inside,color='#4477AA',label='Finished inside window')
        ax[1].bar(x,after,bottom=inside,color='#AA3377',label='Finished after window')
        cc_axis(ax[1],'Tasks started during 30-minute window','(b) Latency sample completion')
        ax[1].legend(fontsize=8)
        line(ax[2],'drain_s','Natural drain',scale=60)
        ax[2].axhline(30,color='#999999',ls='--',label='Measurement length')
        cc_axis(ax[2],'Minutes','(c) Time needed after measurement')
        ax[2].legend(fontsize=8)
        save(fig,'07_measurement_cohorts','Finite-window effects | keep completion and admission cohorts distinct',
             'cc64: all 15 window completions started in warmup; all 15 window admissions finished later.\nAll admission-cohort latencies are observed through natural drain; no unfinished task is silently discarded.')
    return figures


def report(rows,n_common,figures,config,paired,series,http_checks):
    fields=[('Task throughput (tasks/min)','tasks_per_min',2),('Completed in window','completed_tasks',0),
        ('Admitted in window / latency n','admitted_tasks',0),('Unique admitted traces','unique_admitted_traces',0),
        ('Task latency mean (s)','task_mean_s',2),('Task latency p50 (s)','task_p50_s',2),('Task latency p95 (s)','task_p95_s',2),
        ('Output throughput: full window (tokens/s)','output_tokens_per_window_s',2),
        ('Output throughput: active backend (tokens/s)','output_tokens_per_active_s',2),
        ('Decode throughput: active decode (tokens/s)','decode_tokens_per_active_s',2),
        ('LLM calls started/s','llm_calls_started_per_s',3),('Tool calls started/s','tool_calls_started_per_s',3),
        ('LLM latency mean (s)','llm_mean_s',3),('LLM latency p95 (s)','llm_p95_s',3),
        ('Client TTFT p95 (s)','ttft_p95_s',3),('Client TPOT estimate mean (ms/token)','tpot_mean_ms',2),
        ('Initial backend queue p95 (s)','queue_p95_s',5),('Scheduled to first token p95 (s)','scheduled_to_first_token_p95_s',3),
        ('Tool latency p95 (s)','tool_p95_s',3),('GPU busy: four-card mean (%)','gpu_busy_mean_percent',2),
        ('GPU memory busy: four-card mean (%)','gpu_memory_busy_mean_percent',2),
        ('Allocated memory per GPU mean (GiB)','gpu_allocated_mean_gib',2),('Four-GPU power sum (W)','gpu_power_sum_w',1),
        ('Backend active time (%)','backend_active_percent',2),('Backend running requests mean','backend_running_mean',2),
        ('Backend waiting requests mean','backend_waiting_mean',2),('Backend waiting requests max','backend_waiting_max',0),
        ('KV occupancy mean (%)','kv_mean_percent',2),('KV occupancy max (%)','kv_max_percent',2),
        ('Prefix lookup hit ratio (%)','prefix_hit_percent',2),
        ('Prompt tokens actually reused (%)','prompt_reuse_percent',2),
        ('Lookup tokens / processed prompt tokens','prefix_query_amplification',2),
        ('Prefill compute tokens / sampled second','prefill_compute_tokens_per_sample_s',2),
        ('Preemptions (sample delta)','preemptions_sample_delta',0),
        ('Fully cached last-token recomputes (sample delta)','recomputed_tokens_sample_delta',0),
        ('Replay-node CPU mean (%)','replay_cpu_mean_percent',2),('Inference-node CPU mean (%)','inference_cpu_mean_percent',2),
        ('Native tool errors','native_tool_errors',0),('New tool errors','new_tool_errors',0),
        ('New tool error share (%)','new_tool_error_percent',3),('Cohort tasks finishing after window','cohort_finished_after_window',0),
        ('Warmup tasks completing in window','warmup_completed_in_window',0),
        ('Window admissions completing in window','cohort_completed_in_window',0),
        ('Setup mean (s)','setup_mean_s',2),('Setup p95 (s)','setup_p95_s',2),
        ('LLM wall-time mean per admitted task (s)','llm_only_mean_s',2),
        ('Tool wall-time mean per admitted task (s)','tool_only_mean_s',2),
        ('Mean LLM calls per admitted trace','mean_trace_llm_calls',2),
        ('Mean output tokens per admitted trace','mean_trace_output_tokens',1),
        ('Sandbox build error log count','build_error_log_count',0),
        ('Natural drain (s)','drain_s',1),('Backend startup (s)','startup_s',1)]
    table=['# mini-SWE / Qwen3.6 scale metrics','',
           '| Metric | '+' | '.join(f'cc{cc}' for cc in CCS)+' |','|---|'+'---:|'*7]
    for label,key,digits in fields:
        table.append('| '+label+' | '+' | '.join(f'{r[key]:.{digits}f}' for r in rows)+' |')
    (OUT/'tables.md').write_text('\n'.join(table)+'\n')
    r16,r32,r64=rows[4:]
    paired_median={cc:float(np.median([p['ratio_to_cc1'] for p in paired if p['cc']==cc])) for cc in CCS}
    early=[s for s in series if s['cc']==64 and s['minute_start']<10]
    late=[s for s in series if s['cc']==64 and s['minute_start']>=20]
    def token_reuse(bins):
        prompts=[s['prompt_tokens_per_s']*s['sampled_span_s'] for s in bins]
        return sum(p*s['prompt_reuse_percent'] for p,s in zip(prompts,bins))/sum(prompts)
    llm_share_increase=100*(r64['llm_only_mean_s']-r32['llm_only_mean_s'])/(r64['task_mean_s']-r32['task_mean_s'])
    md=['# mini-SWE 单推理节点 Scale 分析：Qwen3.6 TP4','',
        '**七档数据完整且计数一致。当前 sweep 的最高观测吞吐在 cc32；cc64 出现真实的后端退化，但 30 分钟窗口不足以给出稳态任务容量。**','',
        '[全部图表 PDF](minisweagent_scale.pdf) · [完整指标表](tables.md) · [汇总 CSV](summary.csv) · [校验记录](checks.json)','',
        '| cc | tasks/min | 输出 tokens/s | 任务 p95（分钟） | TTFT p95（秒） | 实际 prompt token 复用 |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"| {r['cc']} | {r['tasks_per_min']:.2f} | {r['output_tokens_per_window_s']:.1f} | {r['task_p95_s']/60:.2f} | {r['ttft_p95_s']:.2f} | {r['prompt_reuse_percent']:.1f}% |")
    md += ['', '## 性能趋势','',
        f"- cc1→32：吞吐从 {rows[0]['tasks_per_min']:.2f} 升至 {r32['tasks_per_min']:.2f} tasks/min，约 {r32['speedup_vs_cc1']:.2f} 倍；输出率同步升至 {r32['output_tokens_per_window_s']:.1f} tokens/s。增益逐渐递减，延迟持续增长。cc32 是这七档中观测到的峰值，未验证为可持续容量或满足某个延迟 SLO 的最优点。",
        f"- cc32→64：任务吞吐下降 {100*(1-r64['tasks_per_min']/r32['tasks_per_min']):.1f}%，输出率下降 {100*(1-r64['output_tokens_per_window_s']/r32['output_tokens_per_window_s']):.1f}%。任务均值从 {r32['task_mean_s']/60:.2f} 升至 {r64['task_mean_s']/60:.2f} 分钟，p95 从 {r32['task_p95_s']/60:.2f} 升至 {r64['task_p95_s']/60:.2f} 分钟。两个吞吐指标都下降，不能只用窗口内完成任务组成变化来解释。",
        f"- 后端初始排队 p95 从 {r32['queue_p95_s']:.2f} 秒升至 {r64['queue_p95_s']:.2f} 秒，平均 waiting 从 {r32['backend_waiting_mean']:.2f} 升至 {r64['backend_waiting_mean']:.2f} 个请求；客户端 TTFT p95 从 {r32['ttft_p95_s']:.2f} 升至 {r64['ttft_p95_s']:.2f} 秒。平均 running 仅 {r32['backend_running_mean']:.2f}→{r64['backend_running_mean']:.2f}，增加的并发主要表现为等待。客户端和后端统计 cohort 不同，不将它们的 p95 相减或相加解释耗时。",
        f"- 同期平均任务生命周期增加量的 {llm_share_increase:.1f}% 来自 LLM 请求区间；工具 p95 仅 {r32['tool_p95_s']:.2f}→{r64['tool_p95_s']:.2f} 秒，setup 均值 {r32['setup_mean_s']:.2f}→{r64['setup_mean_s']:.2f} 秒。现有时间分解指向 LLM 路径的退化；这是墙钟时间归属，不能当作 GPU kernel 时间或独立的因果证明。",
        f"- 相同前 {n_common} 条任务的完整生命周期，相对各自 cc1 的延迟中位倍数在 cc16/32/64 为 {paired_median[16]:.2f}/{paired_median[32]:.2f}/{paired_median[64]:.2f}。同一批任务也明显变慢。但该辅助比较包含 warmup，不代表测量窗口内的配对稳态结果。",
        '', '## 缓存与窗口内变化','',
        f"- 实际 prompt token 复用：cc16/32/64 为 {r16['prompt_reuse_percent']:.2f}%/{r32['prompt_reuse_percent']:.2f}%/{r64['prompt_reuse_percent']:.2f}%；查询命中率分别为 {r16['prefix_hit_percent']:.2f}%/{r32['prefix_hit_percent']:.2f}%/{r64['prefix_hit_percent']:.2f}%。cc64 查询 token / 实际处理 prompt token 达 {r64['prefix_query_amplification']:.2f}×，因此不能把查询口径的 17.8% 当作实际复用率。全部七档均以 source=local_compute/local_cache_hit 原始计数交叉核对，external transfer 为 0。",
        f"- cc32→64，本地 prefill 计算 token 计数率由 {r32['prefill_compute_tokens_per_sample_s']:.0f} 升至 {r64['prefill_compute_tokens_per_sample_s']:.0f} tokens/s，约 {r64['prefill_compute_tokens_per_sample_s']/r32['prefill_compute_tokens_per_sample_s']:.2f} 倍；GPU busy 均值仍为 {r32['gpu_busy_mean_percent']:.1f}%/{r64['gpu_busy_mean_percent']:.1f}%。可确认输入计算 token 增多、输出产出降低，并非 GPU 已经不工作；仍不能仅凭这些计数确定缓存失效的唯一原因。",
        f"- KV 非空闲块占用均值由 {r32['kv_mean_percent']:.1f}% 降至 {r64['kv_mean_percent']:.1f}%，七档引擎抢占计数增量均为 0。KV gauge 不包括 free queue 中仍可复用、但可随时淘汰的历史前缀；总分配显存约 {r64['gpu_allocated_mean_gib']:.1f} GiB/卡也不是活跃 KV 使用量。不能据此判定 OOM，也不能凭低 KV gauge 排除历史缓存淘汰。定义和 vLLM 0.19.1 源码依据见[缓存指标复核](../2026-09-08-openclaw-qwen36-scale/cache_interpretation.md)。",
        f"- cc64 前 10 分钟与最后 10 分钟：每分钟输出率均值约 {np.mean([s['output_tokens_per_s'] for s in early]):.0f}→{np.mean([s['output_tokens_per_s'] for s in late]):.0f} tokens/s；平均 waiting 约 {np.mean([s['backend_waiting_mean'] for s in early]):.1f}→{np.mean([s['backend_waiting_mean'] for s in late]):.1f}；按 token 加权的复用比例约 {token_reuse(early):.1f}%→{token_reuse(late):.1f}%。全窗 45.6% 不能代表末段状态，后者的复用已经很低。分段计数采用各 60 秒 bin 内样本增量，边界约少 1 秒/bin。",
        '- 与 OpenClaw 同配置 sweep 对照：OpenClaw 的最高观测吞吐在 cc16，mini-SWE 在 cc32，明显退化分别出现在更高档位。两类负载不能共用一个“最佳 cc”。这两次都是单轮结果；任务结构、实际覆盖和缓存历史不同，不能据此把差异单独归因于某个 trace 特征。',
        '', '## 数据是否有问题','',
        '- **完整性与计数无异常。** 合计 351 次任务全部完成，报告节点集合与原 trace 对齐，每次 LLM call 的实际输出 token 数等于目标值；客户端、任务报告与后端总生成计数一致。七档 measurement valid，GPU 采样正常且无 error sample，admission cohort 无未完成样本。全部任务日志未发现 `Error building image`，本次有效点没有记录到 sandbox 构建失败重试。',
        '- **正确选用补跑数据。** cc1/2/4/8 来自原 run；cc16/32/64 来自 recovery。排除原 run 中因 OCI 拉取失败而中止、没有完成 measurement 的 cc16。模型和 serve 配置、61 条 trace pool、seed、窗口及 replay 参数一致；唯一 profile 差异为 sandbox build 尝试上限 1→3，仅影响 setup，已记入配置和校验记录。',
        f"- **cc64 的完成数不代表已进入稳态。** 窗口内完成 {r64['completed_tasks']} 条，全都在 warmup 启动；测量窗口中新启动 {r64['admitted_tasks']} 条，全都在窗口结束后完成。自然排空用了 {r64['drain_s']/60:.2f} 分钟，超过 30 分钟测量窗。吞吐 0.50 tasks/min 是准确的该窗口完成率，但不足以断言长期容量只有 0.50。图 07 单独展示两类 cohort。",
        '- **窗口内实际任务组成不同。** 七档 admission cohort 没有共同 trace；cc1 与 cc8 的 admission trace 集合也不相交。cc1 延迟只有 7 个样本，p95 很不稳定。任务延迟完整保留窗口后完成部分，没有做完成样本筛选；初始 8 条任务只用于辅助敏感性分析。',
        f"- **存在少量工具结果变化。** 窗口内原成功→重放失败为 {[r['new_tool_errors'] for r in rows]} 次，占各档工具调用 {min(r['new_tool_error_percent'] for r in rows):.3f}%–{max(r['new_tool_error_percent'] for r in rows):.3f}%。大多数 native error 在采集时就已经是 error，例如非零退出的测试命令；不能把 native error 总数当成新增故障数。转换表也保留原失败→重放成功的情况。采集/回放错误标志一致不代表输出内容一致，当前 replay report 未保留足以逐条诊断的 native stderr；不能据此宣称 SWE 任务解题成功。",
        f"- **cc64 有额外 HTTP 探测流量。** 整个运行日志出现 {http_checks[-1]['http_404_all_run']} 个 HTTP 404，其首条位于日志时间 {http_checks[-1]['first_404_previous_log_time_utc']} 与 {http_checks[-1]['first_404_next_log_time_utc']} 之间，接近测量窗口末尾（结束时间 {http_checks[-1]['measurement_end_utc']}）。日志包含与任务无关的扫描路径；之前各档无 404。全部 chat-completions POST 恰好对应已记录 LLM calls，且均返回 200。性能下降更早已经出现，因此这些探测不能解释退化起点；现有记录不能量化其对末段 API 延迟的影响，下一轮应隔离此类流量。详见 [HTTP 日志计数](http_log_checks.csv)。",
        '', '## 测量口径与后续实验','',
        '- Qwen/Qwen3.6-35B-A3B，vLLM 0.19.1，BF16、TP4、4×A100 40GB；context=262144、max_num_seqs=64、max_num_batched_tokens=2048、GPU memory utilization=0.90、APC 与 thinking 开启。每点新启后端，预热 120 秒，测量 1800 秒，自然排空；seed=42。cc 包含 setup、replay、cleanup。bash 在原生 Singularity sandbox 中执行，工具 timeout=60 秒。',
        '- 这是固定 trace 的性能重放：LLM 输出长度按录制 token 数控制，工具动作按 trace 执行，并不让新模型自由决定下一步。因此结果描述该负载的重放性能，不是 Qwen 重新解题的准确率或完整 agent 效果。',
        '- tasks/min = 测量窗口内完成任务数 / 30 分钟；任务 latency = 测量窗口内启动任务的完整生命周期。LLM/tool latency 按窗口内启动的调用统计，后端 latency 按窗口内首次调度的请求统计，各 cohort 不必相同。',
        '- 主图 output tokens/s 使用完整 1800 秒分母；active-output 和 decode throughput 使用不同活跃时间分母，仅列在指标表。时间堆叠使用互斥区间并集，LLM/tool 重叠单列，分项之和等于任务均值。',
        '- actual prompt reuse = 同采样边界 Δcached prompt / Δprompt；cache lookup ratio = Δhits / Δqueries。新增计数器采用窗内首末样本差，跨度约 1799 秒。`prompt_tokens_recomputed_total` 在此版本主要记录完全缓存时强制计算末 token 的特殊情况，其为 0 不能排除缓存 miss 后重新计算历史输入。客户端 TPOT 来自流式 chunk 估算；scheduled→first token 是墙钟间隔，不是纯 prefill kernel 时间。',
        '- 下一轮优先复核 cc32/64，并可在两者之间增加 cc48 定位退化区间。以请求队列、token 输出率和缓存复用的时间曲线判断是否稳定；cc64 任务生命周期已达一小时量级，应显著延长 warmup/measurement，或另做固定相同 trace 集合、全部完成的 batch 实验。后者报告 batch makespan/吞吐，不能混入本报告的固定窗口主图。',
        '- 在转为论文中的容量结论前，需要独立重复，以及相同 trace 组成的对照。当前不把一轮中的任务当成独立实验重复来构造置信区间。工具结果变化应另外检查；无需因已录制的测试失败而将整档性能结果作废。',
        '', '## 图表','']
    for name in figures:
        md += [f'### {name}','',f'![{name}]({name}.png)','',f'[矢量 PDF]({name}.pdf)','']
    md += ['## 数据与复现','',
        '- `summary.csv` / `tables.md`：七档指标；`tasks.csv`：351 次任务、cohort 标记及完整时间分解。',
        '- `initial_eight_paired_latency.csv`：每档相同前 8 条任务的辅助比较，保留 admission phase。',
        '- `tool_error_transitions.csv`：工具采集/回放错误标志的四种转换；`timeseries_60s.csv`：210 个分钟区间的后端与完成数趋势。',
        '- `checks.json`：计数、source token 交叉核对与构建错误日志计数；`http_log_checks.csv`：服务日志 HTTP 状态检查。',
        '- `run_config.json`：配置和输入选择；原始输入、工作区及大日志保留在本地，不包含在报告中。','',
        '```bash','.venv/bin/python reports/2026-09-13-minisweagent-qwen36-scale/analyze.py','```','',
        f'输入基目录：`{RUN}`。具体七档路径见 `run_config.json` 的 `analysis_provenance.points`。脚本只读取既有数据，不调用推理后端或重放工具。','']
    (OUT/'README.md').write_text('\n'.join(md))
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')


def main():
    rows,tasks,series,config=extract()
    paired,n_common=paired_tasks(tasks)
    http_checks=audit_http_logs()
    figures=plot(rows,tasks,series,paired,n_common)
    report(rows,n_common,figures,config,paired,series,http_checks)
    print(f'Wrote {len(figures)} figures; all seven points passed checks; initial paired set n={n_common}.',flush=True)


if __name__ == '__main__':
    main()
