"""Inventory captured traces and draw an initial workload-characterization figure.

Use the full preflight candidate pools, retaining replay eligibility as a label.
No inference, tool execution, or modifications to captured data are performed.
"""
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PREFLIGHT = REPO / 'runs/scaling/preflight-20260906-1'
OPENCLAW_SOURCE = REPO / 'runs/openclaw-gaia/fixed-first30-7576424/tasks'


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


tasks, calls, tools, inventory = [], [], [], []
for workload in ('openclaw', 'minisweagent'):
    pf = read(PREFLIGHT / workload / 'preflight.json')
    for record in pf['traces']:
        path = Path(record['trace_path'])
        trace = read(path)
        llm = [n for n in trace['nodes'] if n['type'] == 'llm']
        tool_nodes = [n for n in trace['nodes'] if n['type'] == 'tool']
        token_rows = {n['node_id']: n for n in record['llm_calls']}
        if workload == 'openclaw':
            with (OPENCLAW_SOURCE / path.stem / 'trajectory.jsonl').open() as handle:
                events = [json.loads(line) for line in handle]
            responses = [e['message'] for e in events if e.get('type') == 'message'
                         and e['message'].get('role') == 'assistant']
            terminal = responses[-1].get('stopReason')
        else:
            trajectory = read(path.parent / 'trajectory.json')
            # One provider response is stored on a user-role event; include it.
            responses = [m for m in trajectory['messages']
                         if isinstance(m.get('extra', {}).get('response'), dict)]
            terminal = trajectory['info'].get('exit_status')
        assert len(llm) == len(responses) == len(token_rows)
        task_calls = []
        for index, (node, response) in enumerate(zip(llm, responses), 1):
            output = node['output_tokens']
            if workload == 'openclaw':
                usage = response['usage']
                original_output = usage['output']
                original_input = usage['input']
                reasoning = usage.get('reasoningTokens')
                response_id = response.get('responseId')
            else:
                native = response['extra']['response']
                usage = native['usage']
                original_output = usage['completion_tokens']
                original_input = usage['prompt_tokens']
                reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                response_id = native.get('id')
            known = original_output == output and isinstance(reasoning, int)
            if known:
                assert 0 <= reasoning <= output
            elif original_output != output:
                assert workload == 'openclaw' and response.get('stopReason') == 'aborted'
            row = dict(workload=workload, trace_id=trace['trace_id'], node_id=node['id'],
                call_index=index, trace_llm_calls=len(llm), replay_32k_eligible=record['eligible'],
                input_qwen_tokens=token_rows[node['id']]['prompt_tokens'],
                input_nemotron_usage_tokens=original_input if original_output == output else None,
                output_recorded_tokens=output, output_reasoning_tokens=reasoning if known else None,
                output_nonreasoning_tokens=output-reasoning if known else None,
                output_unknown_tokens=0 if known else output, response_id=response_id)
            calls.append(row)
            task_calls.append(row)
        for node in tool_nodes:
            tools.append(dict(workload=workload, trace_id=trace['trace_id'], node_id=node['id'],
                replay_32k_eligible=record['eligible'], tool_name=node['request']['name'],
                recorded_error=node['recorded_result']['isError']))
        tasks.append(dict(workload=workload, trace_id=trace['trace_id'],
            trace_path=str(path), replay_32k_eligible=record['eligible'], terminal_status=terminal,
            llm_calls=len(llm), tool_calls=len(tool_nodes),
            unique_tool_names=len({n['request']['name'] for n in tool_nodes}),
            input_qwen_total_tokens=sum(c['input_qwen_tokens'] for c in task_calls),
            max_input_qwen_tokens=max(c['input_qwen_tokens'] for c in task_calls),
            output_recorded_total_tokens=sum(c['output_recorded_tokens'] for c in task_calls),
            output_reasoning_known_tokens=sum(c['output_reasoning_tokens'] or 0 for c in task_calls),
            output_nonreasoning_known_tokens=sum(c['output_nonreasoning_tokens'] or 0 for c in task_calls),
            output_unknown_tokens=sum(c['output_unknown_tokens'] for c in task_calls)))
    subset = [t for t in tasks if t['workload'] == workload]
    inventory.append(dict(workload=workload, traces=len(subset),
        replay_eligible=sum(t['replay_32k_eligible'] for t in subset),
        llm_calls=sum(t['llm_calls'] for t in subset), tool_calls=sum(t['tool_calls'] for t in subset),
        median_llm_calls=median(t['llm_calls'] for t in subset),
        median_tool_calls=median(t['tool_calls'] for t in subset),
        output_recorded_total_tokens=sum(t['output_recorded_total_tokens'] for t in subset)))
write_csv('tasks.csv', tasks)
write_csv('llm_calls.csv', calls)
write_csv('tool_calls.csv', tools)
write_csv('inventory.csv', inventory)
print(json.dumps(inventory, indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
styles = [('openclaw', 'OpenClaw / GAIA', '#2271B2'),
          ('minisweagent', 'mini-SWE / SWE-bench', '#D55E00')]
for workload, label, color in styles:
    for ax, records, key in [(axes[0,0],tasks,'llm_calls'), (axes[0,1],tasks,'tool_calls'),
                             (axes[0,2],calls,'input_qwen_tokens'), (axes[1,0],calls,'output_recorded_tokens')]:
        values = sorted(r[key] for r in records if r['workload'] == workload)
        ax.step(values, [(i+1)/len(values) for i in range(len(values))], where='post',
                color=color, label=label)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.2)
for ax, label in [(axes[0,0],'LLM calls / trace'), (axes[0,1],'Tool calls / trace'),
                  (axes[0,2],'Prompt tokens / call (Qwen tokenizer)'),
                  (axes[1,0],'Recorded output tokens / call (Nemotron)')]:
    ax.set(xlabel=label, ylabel='ECDF')
axes[0,0].legend(fontsize=8, loc='lower right')
axes[0,2].axvline(32768, color='gray', linestyle=':', linewidth=1)
axes[0,2].text(32768, .2, ' 32K prompt reference', rotation=90, fontsize=8)
for i, (workload, label, _) in enumerate(styles):
    selected = [c for c in calls if c['workload'] == workload]
    total = sum(c['output_recorded_tokens'] for c in selected)
    left = 0
    for key, title, color in [('output_reasoning_tokens','Reasoning','#4477AA'),
                               ('output_nonreasoning_tokens','Non-reasoning','#66CCEE'),
                               ('output_unknown_tokens','Unknown','#BBBBBB')]:
        fraction = 100*sum(c[key] or 0 for c in selected)/total
        axes[1,1].barh(i,fraction,left=left,color=color,label=title if i==0 else None)
        if fraction>5:
            axes[1,1].text(left+fraction/2,i,f'{fraction:.1f}%',ha='center',va='center',fontsize=9)
        left += fraction
axes[1,1].set(yticks=[0,1],yticklabels=['OpenClaw','mini-SWE'],xlabel='Share of recorded output tokens (%)',xlim=(0,100))
axes[1,1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3)
tool_colors = dict(web_search='#4477AA',web_fetch='#66CCEE',exec='#228833',read='#CCBB44',bash='#D55E00')
for i,(workload,label,_) in enumerate(styles):
    counts=Counter(t['tool_name'] for t in tools if t['workload']==workload)
    left=0
    for name,count in counts.items():
        fraction=100*count/sum(counts.values())
        axes[1,2].barh(i,fraction,left=left,color=tool_colors[name],label=name)
        if fraction>8:axes[1,2].text(left+fraction/2,i,f'{fraction:.1f}%',ha='center',va='center',fontsize=9)
        left+=fraction
axes[1,2].set(yticks=[0,1],yticklabels=['OpenClaw','mini-SWE'],xlabel='Tool API calls (%)',xlim=(0,100))
axes[1,2].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3)
fig.suptitle('Captured trace characterization: full candidate pools\n24 OpenClaw traces; 22 mini-SWE traces (before context filtering)',fontsize=13)
fig.text(.5,.01,'Prompt counts describe Qwen replay inputs; output counts/usage describe Nemotron capture. Tool API names do not describe bash semantics.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.92))
fig.savefig(OUT/'preview.png',dpi=170)
fig.savefig(OUT/'preview.pdf')
plt.close(fig)
