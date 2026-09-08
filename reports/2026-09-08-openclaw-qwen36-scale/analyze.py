"""Analyze the seven completed OpenClaw points; never run inference/tools.

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
            samples.append({'time': stamp, **{k: float(v) for k,v in found.items()}})
    assert len(samples) > 1700
    first, last = samples[0], samples[-1]
    counters = {k: last[k]-first[k] for k in PROM_NAMES if k.endswith('_total')}
    assert all(v >= 0 for v in counters.values())
    bins = []
    for i in range(30):
        left, right = start+i*60, start+(i+1)*60
        selected = [s for s in samples if left <= s['time'] < right]
        a,b = selected[0], selected[-1]
        dt = b['time']-a['time']
        delta = lambda key: b[key]-a[key]
        queries = delta('prefix_cache_queries_total')
        bins.append(dict(cc=cc, minute_start=i, minute_end=i+1, samples=len(selected),
            sampled_span_s=dt,
            output_tokens_per_s=delta('generation_tokens_total')/dt,
            prompt_tokens_per_s=delta('prompt_tokens_total')/dt,
            prefix_hit_percent=100*delta('prefix_cache_hits_total')/queries if queries else None,
            preemptions=delta('num_preemptions_total'),
            recomputed_tokens=delta('prompt_tokens_recomputed_total'),
            backend_running_mean=float(np.mean([s['num_requests_running'] for s in selected])),
            backend_waiting_mean=float(np.mean([s['num_requests_waiting'] for s in selected])),
            kv_mean_percent=100*float(np.mean([s['kv_cache_usage_perc'] for s in selected])),
            completed_tasks=sum(left <= t['ended_unix'] < right for t in tasks)))
    return counters, bins, last['time']-first['time']


def extract():
    summaries, task_rows, errors, checks, series = [], [], [], [], []
    config = read(RUN / 'config.json')
    baseline = read(RUN / 'experiment/openclaw-cc1/benchmark/manifest.json')
    profile = read(Path(baseline['profile_path']))
    assert profile['tool_executor']['config']['web_search_mode'] == 'recorded_delay'
    traces = {}
    for cc in CCS:
        point = RUN / 'experiment' / f'openclaw-cc{cc}'
        root = point / 'benchmark'
        s, manifest = read(root/'summary.json'), read(root/'manifest.json')
        m, b = s['measurement'], s['measurement']['backend_throughput']
        start,end = m['started_unix'], m['ended_unix']
        assert s['status'] == 'completed' and m['valid'] and not s['dry_run']
        assert end-start == 1800 and manifest['concurrency'] == cc
        assert read(point/'service.json')['config'] == config['serve']
        for key in ['trace_pool','seed','warmup_seconds','duration_seconds','llm_replay','concurrency_scope','profile_path']:
            assert manifest[key] == baseline[key], (cc,key)
        assert len(manifest['trace_pool']) == 110
        assert m['incomplete_admission_cohort'] == 0 and b['status'] == 'measured'
        for key in ['backend_client_counts_match','counter_counts_match','inference_gpu_samples_ok','window_has_backend_tokens']:
            assert s['validation'][key]
        for key in ['metrics','inference_metrics']:
            assert m[key]['error_samples'] == 0
            assert m[key]['gpu_status_counts'] == {'ok':m[key]['samples']}
        total_tokens = total_llm = search_delay = 0
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
            assert {n['node_id'] for n in report['nodes']} == set(original)
            admitted = start <= task['started_unix'] < end
            phases = replay_partition(report)
            cleanup = task['lifecycle_seconds']-report['setup_seconds']-report['replay_makespan_seconds']
            assert cleanup >= -.01
            task_rows.append(dict(cc=cc, task_id=task['task_id'], trace_id=trace['id'], cycle=task['cycle'],
                admission_phase=task['admission_phase'], in_admission_cohort=admitted,
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
                    if source['request']['name'] == 'web_search':
                        assert node['tool_replay']['mode'] == 'recorded_delay'
                        search_delay += 1
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
            preemptions_sample_delta=counters['num_preemptions_total'],
            recomputed_tokens_sample_delta=counters['prompt_tokens_recomputed_total'],
            prompt_tokens_per_sample_s=counters['prompt_tokens_total']/raw_span,
            raw_counter_sample_span_s=raw_span,
            drain_s=s['ended_unix']-end,
            startup_s=read(point/'ready.json')['ready_unix']-read(point/'service.json')['started_unix'])
        for key in ['setup_s','llm_only_s','tool_only_s','overlap_s','replay_gaps_s','cleanup_s']:
            row[key.replace('_s','_mean_s')] = float(np.mean([t[key] for t in cohort]))
        summaries.append(row)
        checks.append(dict(cc=cc, completed_tasks_all=len(s['tasks']), llm_calls_all=total_llm,
            output_tokens_all=total_tokens, recorded_delay_search_calls=search_delay,
            measurement_seconds=1800, validation_passed=True))
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
    groups = defaultdict(list)
    for t in tasks:
        if t['in_admission_cohort']:
            groups[t['cc'],t['trace_id']].append(t['lifecycle_s'])
    common = set.intersection(*[{tid for (c,tid) in groups if c==cc} for cc in CCS])
    result = []
    for trace_id in sorted(common):
        base = float(np.mean(groups[1,trace_id]))
        for cc in CCS:
            latency = float(np.mean(groups[cc,trace_id]))
            result.append(dict(trace_id=trace_id,cc=cc,n_replays=len(groups[cc,trace_id]),
                               mean_lifecycle_s=latency,ratio_to_cc1=latency/base))
    assert result
    write_csv('paired_trace_latency.csv',result)
    return result,len(common)


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
    with PdfPages(OUT/'openclaw_scale.pdf') as pdf:
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
        ax[0,0].set_ylim(0,6.8)
        line(ax[0,1],'task_p50_s','Median',scale=60);line(ax[0,1],'task_p95_s','P95','#EE6677',scale=60)
        cc_axis(ax[0,1],'Task lifecycle (minutes)','(b) Task latency',True);ax[0,1].legend()
        line(ax[0,2],'output_tokens_per_window_s','Window output rate');cc_axis(ax[0,2],'Output tokens / second','(c) Inference output throughput')
        line(ax[1,0],'ttft_p95_s','Client TTFT');line(ax[1,0],'queue_p95_s','Backend initial queue','#EE6677')
        cc_axis(ax[1,0],'P95 seconds','(d) First-token and queue delay',True);ax[1,0].legend(fontsize=8)
        line(ax[1,1],'gpu_busy_mean_percent','GPU busy');line(ax[1,1],'backend_active_percent','Backend active','#228833')
        cc_axis(ax[1,1],'Window mean / time share (%)','(e) Inference utilization');ax[1,1].set_ylim(0,105);ax[1,1].legend(fontsize=8)
        line(ax[1,2],'prefix_hit_percent','Prefix hit rate');cc_axis(ax[1,2],'Prefix tokens hit (%)','(f) Prefix-cache reuse');ax[1,2].set_ylim(0,105)
        save(fig,'01_scale_overview','OpenClaw scaling | Qwen3.6-35B-A3B, TP4, 110-trace pool',
             '120 s warmup + 1800 s measurement; one repetition per cc. Latencies follow window admissions through natural drain.\nweb_search uses recorded delay. cc16 is the best observed point in this sweep, not an SLO capacity bound.')

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
        cc_axis(ax[0,1],'Reported KV occupancy (%)','(b) KV-cache occupancy');ax[0,1].set_ylim(0,100);ax[0,1].legend()
        ax[1,0].bar(x,[r['preemptions_sample_delta'] for r in rows],color='#AA3377')
        cc_axis(ax[1,0],'Preemption counter delta','(c) Engine preemptions')
        if all(r['preemptions_sample_delta']==0 for r in rows):
            ax[1,0].set(ylim=(0,1),yticks=[0,1])
            ax[1,0].plot(x,np.zeros(7),'o',color='#AA3377',clip_on=False)
            ax[1,0].text(.5,.55,'0 in every measured window',transform=ax[1,0].transAxes,ha='center',fontsize=12)
        line(ax[1,1],'tpot_mean_ms','Mean client TPOT estimate');cc_axis(ax[1,1],'Milliseconds / output token','(d) Output streaming slowdown')
        save(fig,'03_backend_pressure','Backend scheduling, cache and output streaming',
             'KV occupancy is the reported vLLM gauge, not total allocated GPU memory. Counters span the in-window samples (~1799 s).\nTPOT is estimated from client streaming chunks; it is not an exact per-token engine measurement.')

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
        cc_axis(ax[1],'Lifecycle / same-trace cc1 mean',f'(b) Same-trace comparison (n={n_common})',True);ax[1].legend(fontsize=8)
        save(fig,'04_latency_distributions','Latency distributions and workload-mix sensitivity',
             'Boxes: median / IQR, whiskers 1.5 IQR; every observed task is shown. No censored task latency is dropped.\nPaired comparison averages repeats per trace, then weights shared traces equally; it does not control cache history or tool outcomes.')

        fig,ax=plt.subplots(2,2,figsize=(12,8))
        for cc,color in [(16,'#4477AA'),(32,'#EE6677'),(64,'#AA3377')]:
            selected=[s for s in series if s['cc']==cc]
            for a,key,label,title in [(ax[0,0],'output_tokens_per_s','Output tokens / second','(a) Output throughput'),
                (ax[0,1],'backend_waiting_mean','Mean waiting requests','(b) Queue accumulation'),
                (ax[1,0],'prefix_hit_percent','Prefix hit rate (%)','(c) Interval cache hit rate'),
                (ax[1,1],'completed_tasks','Tasks completed / minute','(d) Task completions')]:
                a.plot([s['minute_end'] for s in selected],[s[key] for s in selected],color=color,label=f'cc{cc}')
                a.set(xlabel='Minutes since measurement start',ylabel=label,title=title);a.grid(alpha=.18)
        for a in ax.flat:a.legend(fontsize=8)
        save(fig,'05_window_trends','Within-window trends | cc16, cc32 and cc64',
             '60-second bins; token/cache rates use sampled counter deltas within each bin (~59 s). Task counts use full 60 s bins.\nThese curves assess transient behavior; a 30-minute measurement does not by itself establish steady state.')

        fig,ax=plt.subplots(1,3,figsize=(14,5.5))
        line(ax[0],'tool_p95_s','Tool p95');cc_axis(ax[0],'Seconds','(a) Tool latency');ax[0].set_ylim(0,1.2)
        line(ax[1],'new_tool_error_percent','Capture success → replay error','#EE6677')
        cc_axis(ax[1],'Share of tool calls started in window (%)','(b) Newly failing tool calls');ax[1].set_ylim(bottom=0)
        line(ax[2],'unique_admitted_traces','Unique traces admitted');cc_axis(ax[2],'Unique traces out of 110','(c) Admission-cohort coverage');ax[2].set_ylim(0,115)
        save(fig,'06_tools_and_coverage','Tool behavior and sample coverage',
             'Native tool failures are retained; engine validation does not certify identical tool outcomes.\nAll points use the same trace pool and seed, but complete different task cohorts within a fixed window.')
    return figures


def report(rows,n_common,figures,config,paired,series):
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
        ('Prefix-cache hit ratio (%)','prefix_hit_percent',2),('Preemptions (sample delta)','preemptions_sample_delta',0),
        ('Recomputed prompt tokens (sample delta)','recomputed_tokens_sample_delta',0),
        ('Replay-node CPU mean (%)','replay_cpu_mean_percent',2),('Inference-node CPU mean (%)','inference_cpu_mean_percent',2),
        ('Native tool errors','native_tool_errors',0),('New tool errors','new_tool_errors',0),
        ('New tool error share (%)','new_tool_error_percent',3),('Cohort tasks finishing after window','cohort_finished_after_window',0),
        ('Natural drain (s)','drain_s',1),('Backend startup (s)','startup_s',1)]
    table=['# OpenClaw / Qwen3.6 scale metrics','', '| Metric | '+' | '.join(f'cc{cc}' for cc in CCS)+' |',
           '|---|'+'---:|'*7]
    for label,key,digits in fields:table.append('| '+label+' | '+' | '.join(f'{r[key]:.{digits}f}' for r in rows)+' |')
    (OUT/'tables.md').write_text('\n'.join(table)+'\n')
    r16=rows[4];r32=rows[5];r64=rows[6]
    paired_median={cc:float(np.median([p['ratio_to_cc1'] for p in paired if p['cc']==cc])) for cc in CCS}
    late64=[s for s in series if s['cc']==64 and s['minute_start']>=20]
    early64=[s for s in series if s['cc']==64 and s['minute_start']<10]
    md=['# OpenClaw 单推理节点 Scale 分析：Qwen3.6 TP4','',
        '**7 个点均完成并通过测量校验。当前 sweep 的吞吐峰值在 cc16；cc32/64 吞吐下降、延迟和排队上升。**','',
        '[全部图表 PDF](openclaw_scale.pdf) · [完整指标表](tables.md) · [汇总 CSV](summary.csv)','',
        '## 主要观察','',
        f"- 吞吐从 cc1 的 {rows[0]['tasks_per_min']:.2f} tasks/min 上升到 cc16 的 {r16['tasks_per_min']:.2f}，为 {r16['speedup_vs_cc1']:.2f} 倍；cc32/64 分别回落到 {r32['tasks_per_min']:.2f}/{r64['tasks_per_min']:.2f}，较 cc16 下降 {100*(1-r32['tasks_per_min']/r16['tasks_per_min']):.1f}%/{100*(1-r64['tasks_per_min']/r16['tasks_per_min']):.1f}%。这只是本次七档中观测到的最佳点，未进行重复试验或定义延迟 SLO。",
        f"- 后端窗口输出率同样在 cc16 达到 {r16['output_tokens_per_window_s']:.1f} tokens/s，cc32/64 降到 {r32['output_tokens_per_window_s']:.1f}/{r64['output_tokens_per_window_s']:.1f}。下降同时出现在任务吞吐和后端 token 产出上，具体机制还需结合调度、缓存及请求组成进一步分析。",
        f"- 全七档共有的 {n_common} 条 trace 配对后，cc16/32/64 相对各自 cc1 延迟的中位倍数为 {paired_median[16]:.2f}/{paired_median[32]:.2f}/{paired_median[64]:.2f}。同一条任务也明显变慢，说明总体延迟上升不只是因为某些档位选中了更长任务；仍需注意这个小子集、缓存历史与重复次数不同。",
        f"- cc16 → cc32 → cc64 的任务 p95 为 {r16['task_p95_s']/60:.2f} → {r32['task_p95_s']/60:.2f} → {r64['task_p95_s']/60:.2f} 分钟；客户端 TTFT p95 为 {r16['ttft_p95_s']:.2f} → {r32['ttft_p95_s']:.2f} → {r64['ttft_p95_s']:.2f} 秒。初始调度排队 p95 同期升至 {r32['queue_p95_s']:.2f}/{r64['queue_p95_s']:.2f} 秒，表明高并发下明显存在后端等待。不同指标的 p95 不能相加。",
        f"- Prefix-cache token 命中率由 cc16 的 {r16['prefix_hit_percent']:.1f}% 降到 cc32 的 {r32['prefix_hit_percent']:.1f}%、cc64 的 {r64['prefix_hit_percent']:.1f}%；GPU busy 仍约 {r64['gpu_busy_mean_percent']:.1f}%。GPU 保持活跃并不意味着有用输出保持高吞吐。缓存复用降低、队列积压与输出变慢同时出现，支持进一步研究调度/缓存行为；本次 sweep 尚不能单独证明某一种根因。",
        f"- 原始 Prometheus 中的引擎抢占计数增量（cc1→64）为 {[int(r['preemptions_sample_delta']) for r in rows]}；对应 recomputed-prompt-token 计数增量也全部为 0。本次没有观察到抢占计数增加，不能把吞吐下降归因为已证实的抢占、重计算或 OOM。KV 占用和总分配显存也分开呈现。",
        f"- 窗口内存在明显动态变化：cc64 前 10 分钟平均输出率约 {np.mean([s['output_tokens_per_s'] for s in early64]):.1f} tokens/s，最后 10 分钟约 {np.mean([s['output_tokens_per_s'] for s in late64]):.1f}；平均 waiting 从 {np.mean([s['backend_waiting_mean'] for s in early64]):.1f} 升至 {np.mean([s['backend_waiting_mean'] for s in late64]):.1f}。cc32 也有明显波动，因此 30 分钟均值不能直接当作已建立稳态的长期容量。",
        f"- 工具 p95 在七档之间为 {min(r['tool_p95_s'] for r in rows):.2f}–{max(r['tool_p95_s'] for r in rows):.2f} 秒，未呈现与 LLM 相同的延迟恶化。但存在原成功→回放失败的工具调用，已输出逐工具错误转换表；`valid=True` 不代表所有原生工具结果与采集一致。",
        '', '## 配置与测量口径','',
        '- 模型 Qwen/Qwen3.6-35B-A3B，BF16、TP4，4×A100 40GB；context=262144，max_num_seqs=64，max_num_batched_tokens=2048，GPU memory utilization=0.90，prefix caching 开启，thinking 开启。全部 cc 使用相同后端参数。',
        '- OpenClaw 固定 110-trace pool，seed=42；任务 cc 覆盖 setup、完整 replay 和 cleanup。每点新启后端以清空前缀缓存，预热 120 秒、测量 1800 秒，然后停止 admission 并自然排空；每点一次。',
        '- 仅 web_search 使用 recorded-delay 重放；其他工具原生执行。此系列不能直接等同于真实在线 Brave 搜索的端到端表现，也不能与旧模型/旧 trace 池实验的差异全部归因于模型。',
        '- 任务吞吐 = 测量窗口内完成数 / 1800 秒。任务延迟 = 窗口内启动任务的完整生命周期，包含窗口后完成部分。节点延迟按窗口内启动的调用统计，后端延迟按窗口内首次调度的请求统计；这些 cohort 不必相同。',
        f"- cc64 有 {r64['cohort_finished_after_window']}/{r64['admitted_tasks']} 个延迟样本在窗口结束后才完成，排空用了 {r64['drain_s']/60:.1f} 分钟。不能只看窗口内已完成任务来计算 p95；本报告保留完整 admission cohort。固定时长和长任务意味着窗口可能尚有明显瞬态，见时间趋势图。",
        '- 主吞吐图的 token/s 用完整 1800 秒作分母；active-output/decode throughput 是不同活跃时间分母，仅放在完整指标表，不能混用。LLM-only/task breakdown 使用互斥区间，不重复累加并行调用；其中包含等待与网络时间，并非纯 GPU 计算时间。',
        '- `scheduled_to_first_token` 是首次被调度到首 token 的墙钟间隔，不是单独测得的 prefill kernel 时间。TPOT 来自客户端流式 chunk 的估算。原始计数器新增指标采用测量窗内首末样本之差，边界约少 1 秒；不是重新读取全部 backend token 事件。',
        '- 总 pool 和 seed 一致，但各档实际 admission 数、唯一 trace 覆盖及缓存历史不同。任务级总体曲线可能受 workload mix 影响。',
        f'- 提供全部七档共有的 {n_common} 条 trace 的配对延迟比较：每档先平均同一 trace 的重复回放，再按 trace 等权汇总。该子集用于敏感性分析，不能代表完整 pool，也未消除不同缓存历史和工具结果的影响。',
        '- 这是一次探索性 sweep，没有独立重复，因此不提供把同一轮中的任务当独立实验重复的置信区间，也不把 cc16 宣称为已验证的可持续容量上限。','',
        '## 图表','']
    for name in figures:md.extend([f'### {name}','',f'![{name}]({name}.png)','',f'[矢量 PDF]({name}.pdf)',''])
    quick=['| cc | tasks/min | 任务 p95（分钟） | TTFT p95（秒） | Prefix hit |',
           '|---:|---:|---:|---:|---:|']
    for r in rows:
        quick.append(f"| {r['cc']} | {r['tasks_per_min']:.2f} | {r['task_p95_s']/60:.2f} | {r['ttft_p95_s']:.2f} | {r['prefix_hit_percent']:.1f}% |")
    md[6:6]=quick+['']
    md += ['## 数据与复现','',
        '- `summary.csv` / `tables.md`：七档完整指标；`tasks.csv`：每次任务的 cohort 标记、生命周期及分解。',
        '- `paired_trace_latency.csv`：同 trace 的配对比较；`tool_error_transitions.csv`：采集/回放工具错误标志转换计数。',
        '- `timeseries_60s.csv`：每分钟的输出率、缓存命中、队列、抢占及完成数；`checks.json`：计数和必要校验。',
        '- `run_config.json`：本系列配置快照，不包含凭证。原始实验目录保留在本地，未上传大型日志、trace 或工作区。','',
        '```bash','MPLCONFIGDIR=/tmp/agenttrace-matplotlib .venv/bin/python reports/2026-09-08-openclaw-qwen36-scale/analyze.py','```','',
        f'输入目录：`{RUN}/experiment/openclaw-cc*/`。脚本只读本地实验数据，不执行推理、工具或修改运行任务。','']
    (OUT/'README.md').write_text('\n'.join(md))
    # Preserve existing config fields; add no dataset checksum/hash metadata.
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')


def main():
    rows,tasks,series,config=extract()
    paired,n_common=paired_tasks(tasks)
    figures=plot(rows,tasks,series,paired,n_common)
    report(rows,n_common,figures,config,paired,series)
    print(f'Wrote {len(figures)} figures; {n_common} shared traces; all seven points passed checks.',flush=True)


if __name__ == '__main__':
    main()
