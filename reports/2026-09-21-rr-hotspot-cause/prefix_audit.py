"""Offline, token-exact checks of two adjacent requests at RR hotspot onset."""
import json
from collections import Counter
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
MODEL = Path('/lus/eagle/projects/lc-mpi/ZhijingYe/Models/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0')


def match_requests():
    requests = json.loads((OUT/'requests.json').read_text())
    routes = json.loads((OUT/'routing.json').read_text())
    index = {}
    for r in routes:
        index.setdefault((r['workload'], r['destination'], r['target_output_tokens']), []).append(r)
    matched = []
    counts = {'unique': 0, 'ambiguous': 0, 'unmatched': 0}
    for r in requests:
        candidates = [v for v in index.get((r['workload'], r['replica'], r['output']), [])
                      if 0 <= r['queued']-v['started'] < 2 and 0 <= v['ended']-r['finished'] < 2]
        if len(candidates) == 1:
            v = candidates[0]
            matched.append(dict(**r, task_instance_id=v['task_instance_id'], node_id=v['node_id'], home=v['home']))
            counts['unique'] += 1
        else:
            counts['ambiguous' if candidates else 'unmatched'] += 1
    # A unique candidate per backend record need not be unique in reverse.
    claims = Counter((r['task_instance_id'], r['node_id']) for r in matched)
    accepted = [r for r in matched if claims[r['task_instance_id'], r['node_id']] == 1]
    counts['reverse_collisions_excluded'] = len(matched)-len(accepted)
    counts['unique'] = len(accepted)
    matched = accepted
    # Do not permit two backend records to claim the same middleware event.
    assert len({(r['task_instance_id'], r['node_id']) for r in matched}) == len(matched)
    counts.update(completed_requests=len(requests), rule='Same workload, destination and output count; middleware starts 0–2 s before backend queued and ends 0–2 s after backend finished; unique candidates only.')
    (OUT/'matching_audit.json').write_text(json.dumps(counts, indent=2)+'\n')
    (OUT/'matched_requests.json').write_text(json.dumps(matched, separators=(',', ':'))+'\n')
    return matched


def normalized_payload(payload):
    # Router 0.1.15 deserializes the nested envelope into serde_json::Value
    # (default sorted map); strings, including tool argument JSON, stay strings.
    payload = json.loads(json.dumps(payload, sort_keys=True))
    for tool in payload.get('tools', []):
        fn = tool['function']
        # vLLM FunctionDefinition/ChatCompletionToolsParam declaration order.
        fn = {key: fn[key] for key in ('name', 'description', 'parameters') if key in fn}
        tool.clear()
        tool.update(type='function', function=fn)
    for msg in payload['messages']:
        # vLLM 0.19.1 chat_utils accepts historical `reasoning`, then creates
        # reasoning_content for the template. A recorded reasoning_content
        # field alone is not propagated by that parser.
        msg.pop('reasoning_content', None)
        if msg.get('reasoning') is not None:
            msg['reasoning_content'] = msg['reasoning']
        for tc in msg.get('tool_calls', []) or []:
            fn = tc.get('function', tc)
            if isinstance(fn.get('arguments'), str):
                fn['arguments'] = json.loads(fn['arguments'])
    return payload


def main():
    matched = match_requests()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    def fail(message):
        raise ValueError(message)
    env.globals['raise_exception'] = fail
    config = json.loads((MODEL/'tokenizer_config.json').read_text())
    assert config['chat_template'] == (MODEL/'chat_template.jinja').read_text()
    template = env.from_string(config['chat_template'])
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    pairs = []
    for workload, suffix, trace_id, node_ids in [
        ('openclaw', 'r01/00006', '16d825ff-1623-4176-a5b5-42e0f5c2b0ac', ['llm-032', 'llm-033']),
        ('minisweagent', 'r02/00003', 'sqlfluff__sqlfluff-1733', ['llm-045', 'llm-046']),
    ]:
        trace_path = REPO.parent/'datasets/scale-20260908-collected'/workload/'traces'/trace_id/'trace.json'
        trace = json.loads(trace_path.read_text())
        selected, encoded = [], []
        for node_id in node_ids:
            candidates = [r for r in matched if r['workload'] == workload and r['task_instance_id'].endswith(suffix) and r['node_id'] == node_id]
            assert len(candidates) == 1
            r = candidates[0]
            payload = normalized_payload(next(n for n in trace['nodes'] if n['id'] == node_id)['request']['payload'])
            text = template.render(messages=payload['messages'], tools=payload.get('tools'), add_generation_prompt=True,
                                   **payload.get('chat_template_kwargs', {'enable_thinking': True}))
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            assert len(ids) == r['prompt'], (workload, node_id, len(ids), r['prompt'])
            encoded.append(ids)
            selected.append({**r, 'reconstructed_prompt_tokens': len(ids)})
        common = next((i for i, (a, b) in enumerate(zip(*encoded)) if a != b), min(map(len, encoded)))
        previous, current = selected
        assert previous['replica'] == current['replica']
        assert common > previous['cached'] > current['cached']
        pairs.append({'workload': workload, 'trace_id': trace_id, 'trace_path': str(trace_path),
                      'common_prefix_tokens': common, 'aligned_common_prefix_tokens': common//528*528,
                      'gap_after_previous_finished_seconds': current['queued']-previous['finished'],
                      'elapsed_since_previous_cache_hit_seconds': current['queued']-previous['scheduled'],
                      'previously_reused_prefix_no_longer_reused_tokens': previous['cached']-current['cached'],
                      'requests': selected})
    (OUT/'prefix_pairs.json').write_text(json.dumps(pairs, indent=2)+'\n')
    print(json.dumps([{k: v for k, v in p.items() if k != 'requests'} for p in pairs], indent=2))


if __name__ == '__main__':
    main()
