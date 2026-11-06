"""Matched coverage only; paired success uncertainty and explicit cost totals."""
import json
from pathlib import Path
import numpy as np


def summarize(root):
    root=Path(root)
    records=[json.loads(p.read_text()) for p in sorted((root/'test').glob('t*-s*-*.json'))]
    groups={}
    for r in records:
        if r.get('status')=='COMPLETED':groups.setdefault((r['task_index'],r['seed']),{})[r['arm']]=r
    paired=[v for v in groups.values() if set(v)=={'authors','v4'}]
    totals={}
    for arm in ('authors','v4'):
        rows=[v[arm] for v in paired]
        n=len(rows);wins=sum(r['success'] for r in rows)
        totals[arm]=dict(episodes=n,successes=wins,success_percent=100*wins/n if n else None)
        for key in ('policy_calls','inference_seconds','policy_roundtrip_seconds','simulator_seconds',
                    'selector_seconds','wall_seconds','environment_steps','noise_reused_decisions',
                    'horizon_violations','policy_budget_violations','paid_api_calls','external_vlm_calls'):
            totals[arm][key]=sum(r[key] for r in rows)
        totals[arm]['model_input_tokens']=sum(r['model_input_tokens'] for r in rows) if all(
            r['model_input_tokens'] is not None for r in rows) else None
    d=np.array([int(v['v4']['success'])-int(v['authors']['success']) for v in paired])
    helped=int((d==1).sum());harmed=int((d==-1).sum())
    ci=None;cluster_ci=None;p=None
    if len(d):
        rng=np.random.default_rng(9123)
        boots=np.mean(rng.choice(d,size=(20000,len(d)),replace=True),axis=1)*100
        ci=np.quantile(boots,[.025,.975]).tolist()
        tasks={}
        for v,delta in zip(paired,d):tasks.setdefault(v['authors']['task_index'],[]).append(int(delta))
        means=np.array([np.mean(v) for v in tasks.values()])
        cluster=np.mean(rng.choice(means,size=(20000,len(means)),replace=True),axis=1)*100
        cluster_ci=np.quantile(cluster,[.025,.975]).tolist()
        from math import comb
        m=helped+harmed
        p=min(1.,2*sum(comb(m,k) for k in range(min(helped,harmed)+1))/2**m) if m else 1.
    mismatches=sum(v['authors']['initial_fingerprint']!=v['v4']['initial_fingerprint'] or
        v['authors']['first_prediction']!=v['v4']['first_prediction'] for v in paired)
    return dict(completed_cases=len(paired),planned_cases=120,completed_episodes=len(records),planned_episodes=240,
        arms=totals,paired=dict(helped=helped,harmed=harmed,
            difference_pp=float(d.mean()*100) if len(d) else None,
            episode_bootstrap_95_difference_pp=ci,task_cluster_bootstrap_95_difference_pp=cluster_ci,
            mcnemar_exact_two_sided_p=p,initial_state_or_prediction_mismatches=mismatches),
        interpretation='Matched 24-task five-episodes/task pilot; no full paper reproduction or pretraining-unseen claim.',
        horizon='Strict 720 physical actions; no claim that this is the unpublished paper GR00T launch limit.',
        cost='All VLM inference is local inside GR00T; input token counts include image tokens. No paid API.',
        timing='Shared GPU; timings descriptive, with inference and simulator seconds separated.')


if __name__=='__main__':
    import common
    print(json.dumps(summarize(common.RUN),indent=2))
