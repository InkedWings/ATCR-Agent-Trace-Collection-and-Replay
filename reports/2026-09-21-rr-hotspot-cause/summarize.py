"""Summarize and plot existing onset extraction; no inference or cluster calls."""
import bisect
import csv
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/atcr-report-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def read_csv(name):
    with (OUT/name).open() as f:
        return [{k: (v if k in ('workload', 'replica') else float(v) if v else None)
                 for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    requests = json.loads((OUT/'requests.json').read_text())
    by_samples, by_requests, by_steps = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in read_csv('samples.csv'):
        by_samples[r['workload'], r['replica']].append(r)
    for r in read_csv('steps.csv'):
        by_steps[r['workload'], r['replica']].append(r)
    for r in requests:
        by_requests[r['workload'], r['replica']].append(r)
    rows = []
    for (workload, rid), samples in sorted(by_samples.items()):
        samples.sort(key=lambda r: r['seconds'])
        times = [r['seconds'] for r in samples]
        reqs, steps = by_requests[workload, rid], by_steps[workload, rid]
        for start in range(0, 1170, 30):
            end = start+30
            lo, hi = bisect.bisect_left(times, start), bisect.bisect_left(times, end)
            a, b = samples[lo], samples[hi]
            span = b['seconds']-a['seconds']
            cache = b.get('cache_counter', 0)-a.get('cache_counter', 0)
            compute = b.get('compute_counter', 0)-a.get('compute_counter', 0)
            assert cache >= 0 and compute >= 0 and span > 0
            gauges = samples[lo:hi]
            arrivals = [r for r in reqs if start <= r['queued'] < end]
            progress = [r for r in steps if start <= r['seconds'] < end]
            row = dict(workload=workload, replica=rid, start_seconds=start, end_seconds=end,
                       actual_counter_span_seconds=span, arrivals=len(arrivals),
                       completions=sum(start <= r['finished'] < end for r in reqs),
                       unfinished_at_end=sum(r['queued'] <= end < r['finished'] for r in reqs),
                       requested_output_tokens_s=sum(r['output'] for r in arrivals)/30,
                       prompt_mean=st.mean(r['prompt'] for r in arrivals) if arrivals else None,
                       actual_prefix_reuse=cache/(cache+compute) if cache+compute else None,
                       uncached_input_tokens_s=compute/span,
                       output_tokens_s=(b.get('output_counter', 0)-a.get('output_counter', 0))/span,
                       preemptions=b.get('preemptions', 0)-a.get('preemptions', 0),
                       output_progress_interval_ms=1000*st.mean(r['interval'] for r in progress) if progress else None)
            for key in ('running', 'waiting', 'kv', 'cpu', 'gpu_busy', 'gpu_power'):
                row[key] = st.mean(r[key] for r in gauges)
            rows.append(row)
    with (OUT/'wallclock_30s.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    pairs = json.loads((OUT/'prefix_pairs.json').read_text())
    evidence = {'counter_bin_width_seconds': 30, 'phase': 'seconds since warmup starts',
                'preemptions_first_10_minutes': {}, 'onset_30s': [],
                'same_backend_prefix_pairs': pairs}
    for w, hot in [('openclaw', 'r03'), ('minisweagent', 'r01')]:
        evidence['preemptions_first_10_minutes'][w] = {rid: sum(r['preemptions'] for r in rows if r['workload'] == w and r['replica'] == rid and r['end_seconds'] <= 600) for rid in ('r00', 'r01', 'r02', 'r03')}
        evidence['onset_30s'].extend(r for r in rows if r['workload'] == w and r['replica'] == hot and r['start_seconds'] in (300, 330, 360, 390, 420, 450, 480, 510))
    (OUT/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'savefig.dpi': 160})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, constrained_layout=True)
    specs = [('outstanding', 'Running + waiting requests', 1),
             ('actual_prefix_reuse', 'Actual prompt reuse (%)', 100),
             ('uncached_input_tokens_s', 'Uncached input (k tokens/s)', .001),
             ('output_tokens_s', 'Output tokens/s', 1)]
    colors = {'r00': '#4477AA', 'r01': '#228833', 'r02': '#AA66AA', 'r03': '#BB8800'}
    for row_i, (w, hot, title) in enumerate([('openclaw', 'r03', 'OpenClaw'), ('minisweagent', 'r01', 'mini-SWE')]):
        pair = next(p for p in pairs if p['workload'] == w)
        event = pair['requests'][1]['scheduled']/60
        for col_i, (metric, label, factor) in enumerate(specs):
            ax = axes[row_i, col_i]
            for rid in ('r00', 'r01', 'r02', 'r03'):
                points = [r for r in rows if r['workload'] == w and r['replica'] == rid and r['start_seconds'] < 900]
                y = [(r['running']+r['waiting'] if metric == 'outstanding' else r[metric]) for r in points]
                ax.plot([(r['start_seconds']+15)/60 for r in points],
                        [v*factor if v is not None else math.nan for v in y],
                        color='#CC3311' if rid == hot else colors[rid],
                        lw=2.4 if rid == hot else 1.2, alpha=1 if rid == hot else .8,
                        label=rid+(' (hotspot)' if rid == hot else ''))
            ax.axvline(event, color='#555555', lw=1, ls='--')
            ax.set(title=label, xlim=(0, 15), ylim=(0, None))
            if metric == 'actual_prefix_reuse': ax.set_ylim(0, 100)
            if col_i == 0:
                ax.set_ylabel(title, fontsize=12, weight='bold')
                ax.legend(fontsize=8, loc='upper left')
            if row_i == 1: ax.set_xlabel('Minutes since warmup started')
    fig.suptitle('RR onset: equal arrival counts do not preserve equal service capacity\nDashed line: examined same-backend request with lost prefix reuse; 30 s bins', fontsize=14)
    (OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/rr-onset.png')
    fig.savefig(OUT/'figures/rr-onset.pdf')
    plt.close(fig)
    print(json.dumps({'preemptions': evidence['preemptions_first_10_minutes'], 'counter_bins': len(rows),
                      'prefix_pairs': len(pairs), 'figure': 'figures/rr-onset.png'}, indent=2))


if __name__ == '__main__':
    main()
