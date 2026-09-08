"""Characterize frozen capture traces without inference or tool execution.

Run from the repository root with the existing plotting environment. All tables
retain the full frozen collection; figures select the actual Qwen scale pool.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/agenttrace-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
STYLES = [('openclaw', 'OpenClaw / GAIA', '#0072B2'),
          ('minisweagent', 'mini-SWE / SWE-bench', '#D55E00')]
API_COLORS = dict(web_search='#4477AA', web_fetch='#66CCEE', exec='#228833',
                  read='#CCBB44', write='#AA3377', edit='#EE6677',
                  process='#BBBBBB', bash='#D55E00')
SEMANTIC_COLORS = dict(Retrieval='#4477AA', Inspect='#66CCEE', Modify='#228833',
                       Test='#CCBB44', Environment='#AA3377', Submit_diff='#EE6677',
                       Mixed='#999999', Unclassified='#DDDDDD')


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    # File iteration preserves U+2028/U+2029 inside JSON strings.
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(out, name, rows):
    if not rows:
        return
    with (out / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    return None


def semantic(name, command):
    """Conservative, explicitly heuristic, mutually exclusive call categories."""
    native = {'web_search': 'Retrieval', 'web_fetch': 'Retrieval',
              'read': 'Inspect', 'write': 'Modify', 'edit': 'Modify'}
    if name in native:
        return native[name]
    if name not in {'bash', 'exec'}:
        return 'Unclassified'
    patterns = {
        'Retrieval': r'\b(?:curl|wget)\b|requests\.(?:get|post)\(',
        'Inspect': r'\b(?:cat|head|tail|grep|rg|find|ls|pwd|less)\b|\bsed\s+-n|\.read_text\(',
        'Modify': r'\b(?:apply_patch|patch|touch|mkdir|mv|cp|rm)\b|\bsed\s+-i|\.write_text\(|\.write\(|open\([^\n]*,[ ]*[\x27\x22][wa]|(?:^|\n)\s*cat\s*>',
        'Test': r'\b(?:pytest|unittest|tox|runtests|test\.py|manage\.py\s+test)\b',
        'Environment': r'\b(?:pip|pip3|conda|apt-get|apt|make|cmake)\b',
        'Submit_diff': r'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|\bgit\s+(?:diff|status|add|commit)\b',
    }
    matches = [key for key, pattern in patterns.items() if re.search(pattern, command)]
    return matches[0] if len(matches) == 1 else 'Mixed' if matches else 'Unclassified'


def timing(workload, raw, elapsed):
    """Partition chronological event gaps, never sum overlapping tool timers.

    These are phase estimates, including dispatch, waiting and logging. The
    mini-SWE first-call anchor has one-second provider timestamp resolution.
    A missing timestamp invalidates both adjacent gaps; their time remains in
    the residual. Untimed final submit actions likewise stay unattributed.
    """
    llm_gaps, tool_gaps = {}, {}
    previous = first = last = None
    missing = 0
    response_number = 0
    if workload == 'openclaw':
        events = [(e, e['message']) for e in raw if e.get('type') == 'message']
    else:
        events = [(m, m) for m in raw['messages']]
    for event, message in events:
        role = message.get('role')
        response = (message.get('extra') or {}).get('response')
        is_llm = role == 'assistant' if workload == 'openclaw' else isinstance(response, dict)
        is_tool = role == 'toolResult' if workload == 'openclaw' else role == 'tool'
        if workload == 'openclaw':
            stamp = epoch(event.get('timestamp'))
            if role == 'user' and first is None:
                first = previous = stamp
        else:
            stamp = epoch((message.get('extra') or {}).get('timestamp'))
            if is_llm and response_number == 0:
                first = previous = epoch(response.get('created'))
        if not (is_llm or is_tool):
            continue
        if is_llm:
            response_number += 1
        if stamp is None:
            missing += 1
            previous = None
            continue
        if previous is not None:
            gap = stamp - previous
            if gap < -0.001:
                raise ValueError(f'nonmonotonic native events: {gap}')
            gap = max(0.0, gap)
            if is_llm:
                llm_gaps[response_number] = gap
            else:
                call_id = message.get('toolCallId', message.get('tool_call_id'))
                if call_id is not None:
                    tool_gaps[call_id] = gap
        previous = last = stamp
    llm_seconds, tool_seconds = sum(llm_gaps.values()), sum(tool_gaps.values())
    residual = elapsed - llm_seconds - tool_seconds
    if residual < -0.01:
        raise ValueError(f'phase estimates exceed collection envelope by {-residual}s')
    return dict(llm_phase_seconds=llm_seconds, tool_phase_seconds=tool_seconds,
                other_unattributed_seconds=max(0.0, residual),
                timestamped_span_seconds=last-first if first is not None and last is not None else None,
                missing_timestamp_events=missing), llm_gaps, tool_gaps


def extract(dataset, pools):
    tasks, calls, tools = [], [], []
    for workload, _, _ in STYLES:
        preflight = read(pools / workload / 'preflight.json')
        pf = {t['trace_id']: t for t in preflight['traces'] if t['eligible']}
        selected_paths = {Path(p).resolve() for p in (pools / workload / 'traces.txt').read_text().splitlines() if p}
        selected_ids = {read(p)['trace_id'] for p in selected_paths}
        assert selected_ids == set(pf)
        for ordinal, record in enumerate(jsonl(dataset / workload / 'index.jsonl'), 1):
            path = dataset / workload / record['trace_path']
            trace = read(path)
            trace_id = trace['trace_id']
            selected = trace_id in selected_ids
            if selected:
                assert path.resolve() in selected_paths
            llm = [n for n in trace['nodes'] if n['type'] == 'llm']
            tool_nodes = [n for n in trace['nodes'] if n['type'] == 'tool']
            assert len(llm) == record['llm_calls'] and len(tool_nodes) == record['tool_calls']
            source = (Path(trace['context']['captured_workspace_root']).parent
                      if workload == 'openclaw' else Path(record['source_trace_path']).parent)
            status = read(source / 'status.json')
            raw = jsonl(source / 'trajectory.jsonl') if workload == 'openclaw' else read(source / 'trajectory.json')
            responses = ([e['message'] for e in raw if e.get('type') == 'message' and e['message'].get('role') == 'assistant']
                         if workload == 'openclaw' else [m for m in raw['messages'] if isinstance(m.get('extra', {}).get('response'), dict)])
            assert len(llm) == len(responses), trace_id
            phases, llm_gaps, tool_gaps = timing(workload, raw, status['elapsed_seconds'])
            prompt_rows = {n['node_id']: n for n in pf[trace_id]['llm_calls']} if selected else {}
            local_calls = []
            for i, (node, response) in enumerate(zip(llm, responses), 1):
                native = response if workload == 'openclaw' else response['extra']['response']
                usage = native['usage']
                output = node['output_tokens']
                recorded_usage_output = usage.get('output', usage.get('completion_tokens'))
                matched = recorded_usage_output == output
                reasoning = usage.get('reasoningTokens', (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
                known = matched and isinstance(reasoning, int)
                if known:
                    assert 0 <= reasoning <= output
                prompt = prompt_rows.get(node['id'], {}).get('prompt_tokens')
                if selected:
                    assert prompt_rows[node['id']]['target_output_tokens'] == output and prompt > 0
                row = dict(workload=workload, trace_id=trace_id, scale_selected=selected,
                    node_id=node['id'], call_index=i, trace_llm_calls=len(llm),
                    prompt_qwen36_tokens=prompt,
                    input_nemotron_tokens=usage.get('input', usage.get('prompt_tokens')) if matched else None,
                    output_recorded_tokens=output, output_reasoning_tokens=reasoning if known else None,
                    output_nonreasoning_tokens=output-reasoning if known else None,
                    output_unknown_tokens=0 if known else output,
                    capture_llm_phase_seconds=llm_gaps.get(i))
                assert sum(row[k] or 0 for k in ('output_reasoning_tokens', 'output_nonreasoning_tokens', 'output_unknown_tokens')) == output
                calls.append(row)
                local_calls.append(row)
            for node in tool_nodes:
                request, result = node['request'], node['recorded_result']
                command = str(request.get('arguments', {}).get('command', ''))
                tools.append(dict(workload=workload, trace_id=trace_id, scale_selected=selected,
                    node_id=node['id'], tool_name=request['name'], recorded_error=bool(result['isError']),
                    semantic_heuristic=semantic(request['name'], command),
                    capture_tool_phase_seconds=tool_gaps.get(result.get('toolCallId')),
                    command_preview=command[:500]))
            tasks.append(dict(workload=workload, trace_id=trace_id, scale_selected=selected,
                origin=record['origin'], split=record['split'], agent_status=record['agent_status'],
                trace_path=str(path), native_source_path=str(source),
                llm_calls=len(llm), tool_calls=len(tool_nodes), total_nodes=len(trace['nodes']),
                unique_tools=len({n['request']['name'] for n in tool_nodes}),
                capture_wall_seconds=status['elapsed_seconds'], **phases,
                prompt_qwen36_total_tokens=sum(c['prompt_qwen36_tokens'] for c in local_calls) if selected else None,
                max_prompt_qwen36_tokens=max(c['prompt_qwen36_tokens'] for c in local_calls) if selected else None,
                input_nemotron_known_tokens=sum(c['input_nemotron_tokens'] or 0 for c in local_calls),
                input_nemotron_missing_calls=sum(c['input_nemotron_tokens'] is None for c in local_calls),
                output_recorded_total_tokens=sum(c['output_recorded_tokens'] for c in local_calls),
                output_reasoning_known_tokens=sum(c['output_reasoning_tokens'] or 0 for c in local_calls),
                output_nonreasoning_known_tokens=sum(c['output_nonreasoning_tokens'] or 0 for c in local_calls),
                output_unknown_tokens=sum(c['output_unknown_tokens'] for c in local_calls)))
            if ordinal % 20 == 0:
                print(f'Extracted {workload}: {ordinal}', flush=True)
        assert sum(t['scale_selected'] for t in tasks if t['workload'] == workload) == len(pf)
    return tasks, calls, tools


def values(rows, workload, key):
    return np.array([r[key] for r in rows if r['workload'] == workload and r[key] is not None], dtype=float)


def ecdf(ax, rows, key, title, xlabel, scale=1, log=True):
    observed = []
    for workload, label, color in STYLES:
        x = np.sort(values(rows, workload, key) / scale)
        observed.extend(x)
        # Add a zero-probability starting point; empirical CDF has no bin choices.
        ax.step(np.r_[x[0], x], np.r_[0, np.arange(1, len(x)+1)/len(x)],
                where='post', color=color, linewidth=2, label=f'{label} (n={len(x):,})')
    if log:
        if min(observed) > 0:
            ax.set_xscale('log')
            ax.set_xlim(min(observed)/1.15, max(observed)*1.15)
        else:
            # Keep observed zero-call traces at zero, without negative axes.
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(0, max(observed)*1.15)
    ax.set(title=title, xlabel=xlabel, ylabel='Fraction of traces' if rows and 'llm_calls' in rows[0] else 'Fraction of calls', ylim=(0, 1.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.18, which='both')
    ax.legend(fontsize=8, loc='lower right')


def stacked(ax, matrix, labels, colors, ylabel, title, percent=False, unit_scale=1):
    matrix = np.asarray(matrix, dtype=float)
    if percent:
        matrix = matrix / matrix.sum(axis=1, keepdims=True) * 100
    else:
        matrix = matrix / unit_scale
    bottom = np.zeros(2)
    for j, (label, color) in enumerate(zip(labels, colors)):
        bars = ax.bar([0, 1], matrix[:, j], bottom=bottom, width=.52, label=label, color=color,
                      edgecolor='white', linewidth=.5)
        for i, bar in enumerate(bars):
            share = matrix[i, j] / matrix[i].sum()
            visible = share >= (.04 if percent else .07) and (percent or matrix[i,j] >= matrix.sum(axis=1).max()*.035)
            if visible:
                text = f'{matrix[i,j]:.1f}%' if percent else f'{matrix[i,j]:,.1f}'
                ax.text(bar.get_x()+bar.get_width()/2, bottom[i]+matrix[i,j]/2,
                        text, ha='center', va='center', fontsize=9)
        bottom += matrix[:, j]
    ax.set(xticks=[0, 1], xticklabels=['OpenClaw\n(n=110)', 'mini-SWE\n(n=61)'], ylabel=ylabel, title=title)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2)
    if percent:
        ax.set_ylim(0, 100)


def plot_all(out, tasks, calls, tools):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    figures = []
    with PdfPages(out / 'trace_analysis.pdf') as pdf:
        def save(fig, name, title, note):
            fig.suptitle(title, fontsize=15, y=.99)
            fig.text(.5, .015, note, ha='center', va='bottom', fontsize=9)
            fig.tight_layout(rect=(0, .055, 1, .94), h_pad=2.2, w_pad=2.5)
            fig.savefig(out / f'{name}.png', dpi=180, facecolor='white')
            fig.savefig(out / f'{name}.pdf', facecolor='white')
            pdf.savefig(fig, facecolor='white')
            figures.append(name)
            plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, key, title, xlabel, scale in [
            (axes[0,0], 'llm_calls', '(a) Steps / LLM calls', 'LLM calls per trace', 1),
            (axes[0,1], 'tool_calls', '(b) Tool calls', 'Tool calls per trace', 1),
            (axes[0,2], 'total_nodes', '(c) Total recorded nodes', 'LLM + tool nodes per trace', 1),
            (axes[1,0], 'capture_wall_seconds', '(d) Capture elapsed time', 'Collector wall time (minutes)', 60),
            (axes[1,1], 'prompt_qwen36_total_tokens', '(e) Cumulative replay input', 'Qwen3.6 prompt tokens per trace', 1),
            (axes[1,2], 'output_recorded_total_tokens', '(f) Cumulative recorded output', 'Nemotron output tokens per trace', 1)]:
            ecdf(ax, tasks, key, title, xlabel, scale)
        save(fig, '01_trace_distributions', 'Trace distributions | Qwen scale pool: 110 OpenClaw + 61 mini-SWE',
             'Steps = recorded LLM calls. Input sums count repeated history. Capture time includes endpoint waiting and collector overhead.')

        fig, axes = plt.subplots(2, 2, figsize=(11.6, 8))
        for ax, rows, key, title, xlabel in [
            (axes[0,0], calls, 'prompt_qwen36_tokens', '(a) Input length', 'Qwen3.6 prompt tokens per call'),
            (axes[0,1], calls, 'output_recorded_tokens', '(b) Output length', 'Recorded Nemotron output tokens per call'),
            (axes[1,0], calls, 'capture_llm_phase_seconds', '(c) LLM-associated phase (estimate)', 'Seconds per timestamped LLM phase'),
            (axes[1,1], tools, 'capture_tool_phase_seconds', '(d) Tool-associated phase (estimate)', 'Seconds per timestamped tool phase')]:
            ecdf(ax, rows, key, title, xlabel)
        save(fig, '02_call_distributions', 'Call distributions | calls pooled across selected traces',
             'Time panels use event gaps, including orchestration and waiting; missing timestamps are omitted, not filled with zero.\nCalls within a trace are correlated; these curves are descriptive, not independent statistical replicates.')

        token_matrix, output_matrix = [], []
        for w, _, _ in STYLES:
            tt = [t for t in tasks if t['workload'] == w]
            cc = [c for c in calls if c['workload'] == w]
            token_matrix.append([np.mean([t['prompt_qwen36_total_tokens'] for t in tt]),
                                 np.mean([t['output_recorded_total_tokens'] for t in tt])])
            output_matrix.append([sum(c[k] or 0 for c in cc) for k in
                                  ('output_reasoning_tokens', 'output_nonreasoning_tokens', 'output_unknown_tokens')])
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
        token_labels = ['Qwen3.6 prompt', 'Fixed output target']
        stacked(axes[0], token_matrix, token_labels, ['#4477AA', '#EE6677'], 'Million tokens per trace',
                '(a) Mean replay request budget', unit_scale=1e6)
        stacked(axes[1], token_matrix, token_labels, ['#4477AA', '#EE6677'], 'Share of token budget (%)',
                '(b) Replay input / output budget', percent=True)
        stacked(axes[2], output_matrix, ['Reasoning', 'Non-reasoning', 'Unknown'], ['#228833', '#66CCEE', '#BBBBBB'],
                'Share of recorded output tokens (%)', '(c) Nemotron output composition', percent=True)
        for i in range(2):
            axes[1].text(i, 102, f'Output: {100*token_matrix[i][1]/sum(token_matrix[i]):.2f}%', ha='center', fontsize=9)
        axes[1].set_ylim(0, 111)
        save(fig, '03_token_breakdown', 'Token breakdown | denominators stated separately',
             'Replay budget combines Qwen prompt counts with fixed recorded output lengths; it is not a Nemotron input/output ratio.\nOutput composition uses native Nemotron usage; non-reasoning includes actions and text. Shares are token weighted.')

        api_names = [name for name in API_COLORS if any(t['tool_name'] == name for t in tools)]
        api_matrix, semantic_matrix = [], []
        semantic_names = list(SEMANTIC_COLORS)
        for w, _, _ in STYLES:
            tt = [t for t in tools if t['workload'] == w]
            count = Counter(t['tool_name'] for t in tt)
            api_matrix.append([count[n] for n in api_names])
            count = Counter(t['semantic_heuristic'] for t in tt)
            semantic_matrix.append([count[n] for n in semantic_names])
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        stacked(axes[0], api_matrix, api_names, [API_COLORS[n] for n in api_names], 'Share of tool calls (%)',
                '(a) Native tool API mix', percent=True)
        mean_api = np.asarray(api_matrix)/np.array([110, 61])[:, None]
        stacked(axes[1], mean_api, api_names, [API_COLORS[n] for n in api_names], 'Mean calls per trace',
                '(b) Tool calls per trace')
        stacked(axes[2], semantic_matrix, [n.replace('_', ' / ') for n in semantic_names],
                list(SEMANTIC_COLORS.values()), 'Share of tool calls (%)', '(c) Purpose: heuristic classification', percent=True)
        save(fig, '04_tools_breakdown', 'Tool breakdown | one recorded tool node = one call',
             'API counts are exact. Purpose labels use conservative request-text rules; multiple matched purposes become Mixed.\nPython execution without a clear purpose remains Unclassified. Native tool errors are retained.')

        time_keys = ['llm_phase_seconds', 'tool_phase_seconds', 'other_unattributed_seconds']
        time_matrix = [[float(np.mean(values(tasks, w, key))) for key in time_keys] for w, _, _ in STYLES]
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 6))
        time_labels = ['LLM-associated phases', 'Tool-associated phases', 'Other / unattributed']
        colors = ['#4477AA', '#EE6677', '#BBBBBB']
        stacked(axes[0], time_matrix, time_labels, colors, 'Mean collector wall time (minutes)',
                '(a) Mean capture elapsed time', unit_scale=60)
        stacked(axes[1], time_matrix, time_labels, colors, 'Share of collector wall time (%)',
                '(b) Pooled wall-time composition', percent=True)
        save(fig, '05_execution_time_breakdown', 'Capture execution-time breakdown | event-gap estimates',
             'LLM phases include endpoint waiting/retries; tool phases include dispatch and logging. These are not GPU compute times.\nResidual includes setup, cleanup and untimed events. mini-SWE first-call anchor is a provider timestamp (1 s resolution).')

        fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5))
        for w, label, color in STYLES:
            groups = defaultdict(list)
            for call in calls:
                if call['workload'] == w:
                    groups[call['trace_id']].append(call)
            grid = np.linspace(0, 1, 21)
            curves = []
            for cc in groups.values():
                cc.sort(key=lambda c: c['call_index'])
                x = np.linspace(0, 1, len(cc))
                curves.append(np.interp(grid, x, [c['prompt_qwen36_tokens'] for c in cc]))
            q = np.quantile(curves, [.25, .5, .75], axis=0)
            axes[0].plot(grid*100, q[1]/1000, color=color, label=label, linewidth=2)
            axes[0].fill_between(grid*100, q[0]/1000, q[2]/1000, color=color, alpha=.15)
            tt = [t for t in tasks if t['workload'] == w]
            axes[1].scatter([t['llm_calls'] for t in tt], [t['capture_wall_seconds']/60 for t in tt],
                            color=color, label=label, alpha=.65, s=25, edgecolors='none')
        axes[0].set(title='(a) Context growth: median and IQR', xlabel='Relative LLM-call progress (%)', ylabel='Qwen3.6 prompt tokens (thousands)')
        axes[1].set(title='(b) Steps versus capture elapsed time', xlabel='LLM calls per trace', ylabel='Collector wall time (minutes)', xscale='log', yscale='log')
        for ax in axes:
            ax.legend(fontsize=9)
            ax.grid(alpha=.18)
        save(fig, '06_context_and_duration', 'Trace progression and capture duration',
             'Context curves interpolate within each trace; every trace has equal weight at each progress point.\nCapture duration reflects different collection runs and limits, in addition to workload structure.')
    return figures


def summarize(out, all_tasks, all_calls, all_tools):
    tasks = [t for t in all_tasks if t['scale_selected']]
    calls = [c for c in all_calls if c['scale_selected']]
    tools = [t for t in all_tools if t['scale_selected']]
    rows, inventory, breakdown, tool_summary = [], [], [], []
    for w, _, _ in STYLES:
        for scope, tt in [('all_frozen', [t for t in all_tasks if t['workload'] == w]),
                          ('scale_pool', [t for t in tasks if t['workload'] == w])]:
            inventory.append(dict(workload=w, scope=scope, traces=len(tt),
                llm_calls=sum(t['llm_calls'] for t in tt), tool_calls=sum(t['tool_calls'] for t in tt),
                agent_statuses=json.dumps(dict(Counter(t['agent_status'] for t in tt))),
                splits=json.dumps(dict(Counter(t['split'] for t in tt)))))
        for level, rr, keys in [
            ('trace', tasks, ['llm_calls', 'tool_calls', 'total_nodes', 'capture_wall_seconds',
                             'prompt_qwen36_total_tokens', 'max_prompt_qwen36_tokens', 'output_recorded_total_tokens']),
            ('llm_call', calls, ['prompt_qwen36_tokens', 'output_recorded_tokens', 'capture_llm_phase_seconds']),
            ('tool_call', tools, ['capture_tool_phase_seconds'])]:
            for key in keys:
                x = values(rr, w, key)
                q = np.quantile(x, [0, .25, .5, .75, .9, .95, 1])
                rows.append(dict(workload=w, level=level, metric=key, n=len(x), mean=float(np.mean(x)),
                                 **dict(zip(['min', 'q25', 'median', 'q75', 'p90', 'p95', 'max'], map(float, q)))))
        tt = [t for t in tasks if t['workload'] == w]
        for group, keys in [
            ('replay_token_budget', ['prompt_qwen36_total_tokens', 'output_recorded_total_tokens']),
            ('nemotron_output', ['output_reasoning_known_tokens', 'output_nonreasoning_known_tokens', 'output_unknown_tokens']),
            ('capture_wall_time', ['llm_phase_seconds', 'tool_phase_seconds', 'other_unattributed_seconds'])]:
            totals = [sum(t[k] for t in tt) for k in keys]
            for key, total in zip(keys, totals):
                breakdown.append(dict(workload=w, group=group, component=key, total=total,
                    mean_per_trace=total/len(tt), pooled_share=total/sum(totals),
                    mean_trace_share=float(np.mean([t[key]/sum(t[k] for k in keys) for t in tt]))))
        wt = [t for t in tools if t['workload'] == w]
        for name, count in Counter(t['tool_name'] for t in wt).items():
            selected = [t for t in wt if t['tool_name'] == name]
            tool_summary.append(dict(workload=w, tool_name=name, calls=count, call_share=count/len(wt),
                mean_calls_per_trace=count/len(tt), traces_using=len({t['trace_id'] for t in selected}),
                trace_coverage=len({t['trace_id'] for t in selected})/len(tt),
                native_errors=sum(t['recorded_error'] for t in selected),
                native_error_rate=sum(t['recorded_error'] for t in selected)/count))
    for name, rr in [('summary.csv', rows), ('inventory.csv', inventory), ('breakdown.csv', breakdown), ('tool_summary.csv', tool_summary)]:
        write_csv(out, name, rr)
    return tasks, calls, tools, rows, inventory, breakdown


def report(out, dataset, pools, all_tasks, all_calls, all_tools, figures, rows, inventory):
    checks = dict(full_traces=len(all_tasks), selected_traces=sum(t['scale_selected'] for t in all_tasks),
                  full_llm_calls=len(all_calls), full_tool_calls=len(all_tools),
                  selected_llm_calls=sum(c['scale_selected'] for c in all_calls),
                  selected_tool_calls=sum(t['scale_selected'] for t in all_tools),
                  native_response_counts_match=True, preflight_output_targets_match=True,
                  output_partitions_match=True, time_partitions_match=True,
                  source_and_preflight_paths_exist=True,
                  unknown_output_calls=sum(c['output_unknown_tokens'] > 0 for c in all_calls),
                  selected_unknown_output_calls=sum(c['output_unknown_tokens'] > 0 and c['scale_selected'] for c in all_calls),
                  missing_timestamp_events=sum(t['missing_timestamp_events'] for t in all_tasks))
    (out / 'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    lookup = {(r['workload'], r['metric']): r for r in rows}
    text = ['# 扩展 Trace 数据集分析（2026-09-08）', '',
            '主图对齐本次 Qwen3.6 scale 实验池：**OpenClaw 110 条、mini-SWE 61 条**。每条原始 trace 仅统计一次，不重复累计不同 cc 的回放实例。', '',
            '完整冻结集为 OpenClaw 115 条、mini-SWE 61 条；5 条含 `process` 的 OpenClaw trace 没有进入当前 replay pool。CSV 保留全部 176 条并提供 `scale_selected` 标记，图表使用选中集。', '',
            '| 指标（每 trace） | OpenClaw：median [Q25, Q75] | mini-SWE：median [Q25, Q75] |',
            '|---|---:|---:|']
    for key, label, divisor in [('llm_calls', 'Steps / LLM calls', 1), ('tool_calls', 'Tool calls', 1),
                                ('capture_wall_seconds', '采集总耗时（分钟）', 60),
                                ('prompt_qwen36_total_tokens', '累计 Qwen 输入（百万 tokens）', 1e6),
                                ('output_recorded_total_tokens', '累计原始输出（千 tokens）', 1000)]:
        cols = []
        for w, _, _ in STYLES:
            r = lookup[w, key]
            cols.append(f"{r['median']/divisor:,.2f} [{r['q25']/divisor:,.2f}, {r['q75']/divisor:,.2f}]")
        text.append('| '+label+' | '+' | '.join(cols)+' |')
    text += ['', '当前样本的主要差异：', '',
        '- mini-SWE 的 steps 中位数为 77，OpenClaw 为 8；累计 Qwen 输入中位数分别为 1.486M 与 0.155M。两者都约相差 9.6 倍，而每任务最大 prompt 的中位数仅为 40,995 与 30,333，说明交互链长度是总请求负载差异的重要来源。',
        '- 原始输出中已知 reasoning 占比：OpenClaw 60.1%，mini-SWE 31.3%；OpenClaw 另有 4.1% 输出组成未知。',
        '- OpenClaw 原生工具调用主要是 web_search（48.2%）、web_fetch（31.4%）、exec（19.2%）；mini-SWE 全部为 bash，因此单看 API 类型无法区分代码阅读、修改和测试。',
        '- 采集时间中 LLM 相关阶段占池内总耗时约 83.9% / 78.0%，工具相关阶段约 6.4% / 15.2%（OpenClaw / mini-SWE）。这是含等待与框架开销的采集侧估计，不能据此直接推断 Qwen replay 的瓶颈占比。',
        '', '## 图表', '', '[全部图表 PDF](trace_analysis.pdf)。每张图也提供独立 PNG 和矢量 PDF。', '']
    captions = [
        '任务级分布：steps 定义为 LLM calls；同时展示 tool calls、总节点数、采集耗时、累计输入和输出。ECDF 不依赖直方图分箱；横轴使用对数刻度以显示长尾。',
        '调用级分布：Qwen 输入、原始输出，以及有时间戳的 LLM/tool 阶段耗时。调用级曲线按调用汇总；同一 trace 中的调用不独立。',
        'Token breakdown：Qwen 输入 + 固定输出长度表示 replay 请求预算；输出的 reasoning / non-reasoning / unknown 表示 Nemotron 采集 usage 构成。两种口径不混称为原模型的输入输出比。',
        'Tools breakdown：原生 API 调用份额、每 trace 平均调用次数、命令用途的启发式分类。原生 API 统计精确；用途分类仅供初步探索，Mixed/Unclassified 保留不确定性。',
        '执行时间 breakdown：按相邻事件时间戳划分 LLM 阶段、tool 阶段和其他/未归因；提供平均耗时和池内总耗时加权份额。',
        '上下文增长及 steps–耗时关系：每条 trace 按相对 LLM 进度插值，计算等权 median/IQR。不是把所有 calls 混在一起求进度曲线。']
    for name, caption in zip(figures, captions):
        text += [f'### {name}', '', caption, '', f'![{name}]({name}.png)', '', f'[矢量 PDF]({name}.pdf)', '']
    text += ['## 统计口径与限制', '',
        '- 采集模型为 ALCF Nemotron-3-Ultra；Qwen3.6 输入 token 使用此次已完成的 tokenizer preflight。累计输入重复计算每轮传入的历史上下文，表示请求输入负载，不是唯一信息量，也不是实际未命中 cache 的 prefill 量。',
        '- 输出长度来自 trace 的 `output_tokens`。只有原生 usage 的输出总数与 trace 一致、reasoning 字段有效时才拆分；其他输出全部标为 unknown。Non-reasoning 包括工具参数与可见文本，不等于最终答案。未按消息角色近似拆分输入，以免将文本长度近似宣称为模板后的精确 token 构成。',
        f"- 完整集有 {checks['unknown_output_calls']} 次 OpenClaw 调用的输出组成未知；主图选中集有 {checks['selected_unknown_output_calls']} 次。缺失原生输入 usage 在 CSV 中留空，未填零。Qwen preflight 输入覆盖全部选中 calls。",
        '- 时间来自原始采集记录，不是当前 Qwen cc 实验的性能数据。`capture_wall_seconds` 使用 collector 的 elapsed_seconds；OpenClaw 是 agent 子进程耗时，mini-SWE 还覆盖环境准备、收尾和 trace 构建，二者边界并非完全相同。',
        '- OpenClaw 从首条 user 消息的写入时间开始，使用每条消息外层 timestamp 作为事件完成时间。mini-SWE 使用 extra.timestamp；第一轮用 provider response.created 作为近似起点（秒级分辨率，跨端时钟可能有偏差）。后续所有阶段均采用相邻本地事件时间差。',
        '- 每段时间只归给结束该区间的事件，避免并行工具按各自耗时相加导致重复计算。LLM 阶段可包含 endpoint 排队、重试和框架开销；tool 阶段也包含调度、结果包装和记录开销。因此不能把该图解释为纯 GPU 推理时间/纯工具服务时间，或逐工具的独立延迟。',
        '- mini-SWE 有 2 个带 response 的 user-role 事件缺少时间戳；这些事件前后的间隔均不归给已知阶段。最终 Submitted 工具结果没有时间戳，耗时保留在其他/未归因中。其他项还覆盖首段之前、末段之后、环境启动、收尾与未观测间隔。缺失工具耗时不填零。',
        '- 完整但失败、aborted 或 LimitsExceeded 的轨迹均保留；Submitted 不代表通过 SWE-bench 测试。OpenClaw 扩展批次采用约 240 秒上限，少量补采采用 600 秒，上限附近的堆积不能解释为自然任务耗时分布。不同采集批次的 endpoint 状态也不同。',
        '- mini-SWE 混合 dev 与 test，且本次新增 test 按收集顺序停止，不是全 SWE-bench 的随机样本。这里描述当前 workload pool，不做任务总体代表性或显著性宣称。',
        '- 堆叠图的百分比是池内 token/call/time 加权构成；breakdown.csv 同时提供每 trace 比例的等权平均。p90/p95 仅作为样本描述，未提供把调用当独立重复的置信区间。',
        '- 命令用途分类采用脚本中公开的字符串规则，多个用途命中归 Mixed，未明确用途的 Python 执行归 Unclassified；不能直接作为已人工标注的论文结论。', '',
        '## 数据文件与复现', '',
        '- `tasks.csv`：全部冻结 trace 的逐任务指标、来源路径与选中标记。',
        '- `llm_calls.csv` / `tool_calls.csv`：逐调用指标；缺失值为空。工具表保留截断的命令预览，便于复核启发式分类。',
        '- `summary.csv`：选中池各指标的 n、mean、min、Q25、median、Q75、p90、p95、max。',
        '- `inventory.csv`：完整集和 scale 子集的数量、调用数、终止状态与 split。',
        '- `breakdown.csv` / `tool_summary.csv`：构成、工具覆盖率与原生错误率。',
        '- `checks.json`：必要的数量、usage 分解、时间相加和路径检查；未生成或验证校验和。', '',
        '```bash', 'MPLCONFIGDIR=/tmp/agenttrace-matplotlib .venv/bin/python reports/2026-09-08-expanded-trace-analysis/analyze.py', '```', '',
        f'输入数据集：`{dataset}`。', '', f'输入 preflight/pool：`{pools}`。', '',
        '脚本只读取本地 trace/原始轨迹/preflight，写本报告目录；不请求推理、不调用搜索 API、不改动正在执行的实验。', '']
    (out / 'README.md').write_text('\n'.join(text))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=REPO.parent / 'datasets/scale-20260908-collected')
    parser.add_argument('--pools', type=Path, default=REPO / 'runs/scaling/qwen36-expanded-7594555-20260908/pools')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_tasks, all_calls, all_tools = extract(args.dataset, args.pools)
    for name, records in [('tasks.csv', all_tasks), ('llm_calls.csv', all_calls), ('tool_calls.csv', all_tools)]:
        write_csv(args.output, name, records)
    tasks, calls, tools, rows, inventory, breakdown = summarize(args.output, all_tasks, all_calls, all_tools)
    figures = plot_all(args.output, tasks, calls, tools)
    checks = report(args.output, args.dataset, args.pools, all_tasks, all_calls, all_tools, figures, rows, inventory)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
