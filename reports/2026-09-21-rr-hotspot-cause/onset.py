"""Offline RR onset diagnosis: arrivals, actual token progress, resources, workload.

Only existing files are read. Token events after the first 1200 seconds are not
decoded; later finished records are retained to avoid latency truncation.
"""
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict, Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
LIMIT = 1200


def dist(a):
    a = sorted(a)
    def p(q):
        x = (len(a)-1)*q
        i = int(x)
        return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i) if a else None
    return {"count":len(a), "mean":st.mean(a) if a else None, "p50":p(.5), "p95":p(.95), "max":max(a) if a else None}


def save_csv(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def prom(sample):
    wanted={"vllm:num_requests_running":"running", "vllm:num_requests_waiting":"waiting",
            "vllm:kv_cache_usage_perc":"kv", "vllm:num_preemptions_total":"preemptions",
            "vllm:prompt_tokens_by_source_total":"prompt_source",
            "vllm:generation_tokens_total":"output_counter"}
    out=defaultdict(float)
    for line in sample['vllm_prometheus'].splitlines():
        if not line.startswith('vllm:'):continue
        name=line.split('{',1)[0].split(' ',1)[0]
        if name not in wanted:continue
        key=wanted[name]
        if key=='prompt_source':
            if 'source="local_cache_hit"' in line:key='cache_counter'
            elif 'source="local_compute"' in line:key='compute_counter'
            else:continue
        out[key]+=float(line.split()[-1])
    return dict(out)


def extract():
    requests, minute_rows, step_rows, samples, routes = [],[],[],[],[]
    base=REPO/'runs/scaling/multinode'
    sources=[]
    for workload,job in [('openclaw','7642785'),('minisweagent','7642786')]:
        root=next(base.glob(f'{workload}-routing-n4-round_robin-{job}.*'))
        control=json.loads((root/'control/start.json').read_text()); warm=control['start_unix']
        sources.append({'workload':workload,'root':str(root.relative_to(REPO)),'warmup_start_unix':warm,
                        'measurement_offset_seconds':control['measurement_start_unix']-warm,'limit_seconds':LIMIT})
        for backend in sorted((root/'backends').iterdir()):
            rid=backend.name; print(f'Onset {workload} {rid}',flush=True)
            reqs={}; bins=defaultdict(lambda:defaultdict(list)); output=Counter()
            groups=[]; current_ts=None; current=[]; past_limit=False
            with (backend/'backend.jsonl').open('rb') as f:
                for line in f:
                    if b'"request_finished"' in line:
                        r=json.loads(line)
                        queued=r['scheduled_unix']+r['queued_monotonic']-r['scheduled_monotonic']
                        if not warm<=queued<warm+LIMIT:continue
                        assert r['finish_reason'] in ('length','stop'),r['finish_reason']
                        row={'workload':workload,'replica':rid,'request_id':r['request_id'],
                             'queued':queued-warm,'scheduled':r['scheduled_unix']-warm,
                             'first_token':r['first_token_unix']-warm,'finished':r['finished_unix']-warm,
                             'prompt':r['prompt_tokens'],'cached':r['cached_prompt_tokens'],
                             'uncached':r['prompt_tokens']-r['cached_prompt_tokens'],'output':r['output_tokens'],
                             'queue':r['scheduled_monotonic']-r['queued_monotonic'],
                             'prefill':r['first_token_monotonic']-r['scheduled_monotonic'],
                             'decode':r['last_token_monotonic']-r['first_token_monotonic'],
                             'residence':r['finished_unix']-queued}
                        reqs[r['request_id']]=row;requests.append(row)
                    elif not past_limit and b'"tokens"' in line:
                        r=json.loads(line); ts=r['timestamp_unix']-warm
                        if ts>=LIMIT:
                            past_limit=True;continue
                        if ts<0:continue
                        output[int(ts//60)]+=r['output_tokens']
                        if ts!=current_ts:
                            if current:groups.append((current_ts,current))
                            current_ts=ts;current=[]
                        current.append((r['request_id'],r['output_tokens'],r['decode_tokens']))
            if current:groups.append((current_ts,current))
            for r in reqs.values():
                b=bins[int(r['queued']//60)]
                for k in ('prompt','cached','uncached','output','queue','prefill','decode','residence'):b[k].append(r[k])
            prev=None;prev_ids=set()
            for ts,g in groups:
                ids={x[0] for x in g}; b=bins[int(ts//60)]
                if prev is not None and ids & prev_ids:
                    b['ongoing_output_step_gap'].append(ts-prev)
                    step_rows.append({'workload':workload,'replica':rid,'seconds':ts,'interval':ts-prev,
                                      'emitting_requests':len(ids),'output_tokens':sum(x[1] for x in g),
                                      'decode_tokens':sum(x[2] for x in g),
                                      'first_token_count':sum(x[1]-x[2] for x in g),
                                      'prompt_sum_emitting':sum(reqs[x]['prompt'] for x in ids if x in reqs)})
                prev=ts;prev_ids=ids
            with (backend/'inference-metrics.jsonl').open() as f:
                for line in f:
                    r=json.loads(line);ts=r['timestamp_unix']-warm
                    if ts<0:continue
                    if ts>=LIMIT:break
                    hw=r.get('hardware',{}); gpus=hw.get('gpus',[])
                    row={'workload':workload,'replica':rid,'seconds':ts,**prom(r),
                         'cpu':hw.get('cpu_busy_percent'),'cpu_max':hw.get('cpu_max_busy_percent'),
                         'runnable':hw.get('cpu_runnable_processes'),
                         'gpu_busy':st.mean(g['gpu_busy_percent'] for g in gpus) if gpus else None,
                         'gpu_power':st.mean(g['power_watts'] for g in gpus) if gpus else None}
                    samples.append(row);b=bins[int(ts//60)]
                    for k in ('running','waiting','kv','cpu','cpu_max','runnable','gpu_busy','gpu_power'):
                        if row.get(k) is not None:b[k].append(row[k])
            for i,b in sorted(bins.items()):
                row={'workload':workload,'replica':rid,'minute':i,'arrivals':len(b['prompt']),
                     'output_tokens_s':output[i]/60,'prompt_sum':sum(b['prompt']),
                     'requested_output_sum':sum(b['output']),
                     'prefix_reuse':sum(b['cached'])/sum(b['prompt']) if b['prompt'] else None}
                for k,a in b.items():
                    if a:
                        d=dist(a)
                        for stat in ('mean','p95','max'):row[f'{k}_{stat}']=d[stat]
                minute_rows.append(row)
            with (backend/'routing.jsonl').open() as f:
                for line in f:
                    r=json.loads(line)
                    if warm<=r['started_unix']<warm+LIMIT:
                        routes.append({**r,'workload':workload,'started':r['started_unix']-warm,'ended':r['ended_unix']-warm})
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    (OUT/'requests.json').write_text(json.dumps(requests,separators=(',',':'))+'\n')
    (OUT/'routing.json').write_text(json.dumps(routes,separators=(',',':'))+'\n')
    save_csv(OUT/'minutes.csv',minute_rows)
    save_csv(OUT/'samples.csv',samples)
    save_csv(OUT/'steps.csv',step_rows)
    print(f'Wrote {len(requests)} request records, {len(samples)} metric samples, {len(step_rows)} output intervals.',flush=True)


if __name__=='__main__':
    extract()
