"""Offline cache diagnosis from existing logs; does not launch inference."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
BASE = REPO / 'runs/scaling/baseline-validation-7608115-20260913'


def read(path):
    with path.open() as f:
        return json.load(f)


def admissions(budget):
    point = BASE / ('openclaw-warmed-budget%d/openclaw-cc32' % budget)
    measurement = read(point / 'benchmark/summary.json')['measurement']
    rows = []
    with (point / 'backend-scheduler.jsonl').open() as f:
        for line in f:
            e = json.loads(line)
            if e['event'] != 'cache_at_admission':
                continue
            if not measurement['started_unix'] <= e['timestamp_unix'] < measurement['ended_unix']:
                continue
            groups = e['last']['groups']
            assert [g['manager'] for g in groups] == ['FullAttentionManager', 'MambaManager']
            attention = groups[0]['hit_tokens']
            cached = e['last']['cached_tokens']
            assert groups[1]['hit_tokens'] == cached <= attention <= e['prompt_tokens']
            rows.append(dict(request_id=e['request_id'], timestamp_unix=e['timestamp_unix'],
                             prompt=e['prompt_tokens'], attention_hit=attention,
                             cached=cached, mamba_reduction=attention-cached))
    assert len({r['request_id'] for r in rows}) == len(rows)
    prompt = sum(r['prompt'] for r in rows)
    cached = sum(r['cached'] for r in rows)
    attention = sum(r['attention_hit'] for r in rows)
    return dict(source=str(point.relative_to(REPO)), budget=budget, requests=len(rows),
                reduced_requests=sum(r['mamba_reduction'] > 0 for r in rows),
                prompt_tokens=prompt, uncached_tokens=prompt-cached,
                attention_hit_tokens=attention, cached_tokens=cached,
                mamba_reduction_tokens=attention-cached,
                mamba_reduction_fraction_of_uncached=(attention-cached)/(prompt-cached),
                attention_hit_ratio=attention/prompt, actual_hit_ratio=cached/prompt), rows


def main():
    result = {'admission_scope': 'last cache lookup at admission in the 1800 s measurement window',
              'short_ab': [], 'paired_example': [], 'long_runs': []}
    for budget in (2048, 8192):
        aggregate, rows = admissions(budget)
        result['short_ab'].append(aggregate)
        if budget == 8192:
            selected = {r['request_id']: r for r in rows if r['request_id'] in {
                'chatcmpl-9d2065ec9cb6380b-acc87524',
                'chatcmpl-a9ef81818f261b23-881de6ad',
                'chatcmpl-86a3b06f2bdb8687-86268093'}}
            with (REPO / aggregate['source'] / 'backend.jsonl').open() as f:
                for line in f:
                    if 'request_finished' not in line:
                        continue
                    e = json.loads(line)
                    if e['request_id'] in selected:
                        row = selected[e['request_id']]
                        assert e['cached_prompt_tokens'] == row['cached']
                        row.update(output_tokens=e['output_tokens'],
                                   queue_seconds=e['scheduled_monotonic']-e['queued_monotonic'],
                                   scheduled_to_first_token_seconds=e['first_token_monotonic']-e['scheduled_monotonic'])
            result['paired_example'] = sorted(selected.values(), key=lambda r: r['timestamp_unix'])
    sequence = read(REPO / 'runs/single-node-8192-capacity-20260914/sequence.json')
    for item in sequence['sequence'][:4]:
        point = Path(item['output'])
        s = read(point / 'summary.json')
        row = {k: s[k] for k in ('output_tokens_per_second', 'prompt_tokens', 'cached_prompt_tokens',
                                 'uncached_prompt_tokens', 'actual_prefix_reuse', 'valid', 'steady')}
        row.update(name=item['name'], source=str(point.relative_to(REPO)),
                   uncached_prompt_per_output=s['uncached_prompt_tokens']/s['output_tokens'],
                   uncached_prompt_per_second=s['uncached_prompt_tokens']/s['duration_seconds'])
        if item['workload'] == 'openclaw':
            first = last = None
            with (point / 'backends/r00/inference-metrics.jsonl').open() as f:
                for line in f:
                    e = json.loads(line)
                    t = e['timestamp_unix']
                    if t < s['measurement']['measurement_start_unix']:
                        continue
                    if t > s['measurement']['measurement_end_unix']:
                        break
                    counters = {}
                    for metric in e.get('vllm_prometheus', '').splitlines():
                        if metric.startswith(('vllm:num_preemptions_total{', 'vllm:prompt_tokens_recomputed_total{',
                                              'vllm:prompt_tokens_by_source_total{')):
                            key, value = metric.rsplit(None, 1)
                            counters[key] = float(value)
                    if first is None:
                        first = (t, counters)
                    last = (t, counters)
            row['prometheus_sample_span_seconds'] = last[0]-first[0]
            row['prometheus_counter_deltas'] = {k: v-first[1][k] for k, v in last[1].items()}
        result['long_runs'].append(row)
    (OUT / 'evidence.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'short_ab': result['short_ab'], 'long_runs': [
        {k: r[k] for k in ('name', 'uncached_prompt_per_output', 'uncached_prompt_per_second')}
        for r in result['long_runs']]}, indent=2))


if __name__ == '__main__':
    main()
