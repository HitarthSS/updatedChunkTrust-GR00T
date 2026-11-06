"""Finite, paired GR00T N1.5 evaluation with a persistent frozen policy."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Client

import common as c
c.configure()
sys.path.insert(0, str(c.BENCHMARK))
import numpy as np


def rpc(conn, op, **data):
    conn.send(dict(op=op, **data))
    reply = conn.recv()
    if not reply.get('ok'): raise RuntimeError(reply.get('error', str(reply)))
    return reply['result']


def success(info):
    x = np.asarray(info.get('success', False)).reshape(-1)
    return bool(x[0]) if x.size else False


def policy_request(obs, row, seed, decision, step):
    image = np.asarray(obs[c.IMAGE]).copy()
    state = np.concatenate([np.asarray(obs['state.' + p], dtype=np.float32).reshape(d)
        for p, d in zip(c.PARTS, c.DIMS)])
    literal = obs['annotation.human.coarse_action']
    if not isinstance(literal, str) or not literal.strip(): raise RuntimeError('Missing literal instruction')
    observation = {'video.ego_view': image[None], 'annotation.human.coarse_action': np.array([literal])}
    for p, d in zip(c.PARTS, c.DIMS):
        observation['state.' + p] = np.asarray(obs['state.' + p], dtype=np.float32).reshape(1, d)
    return dict(observation=observation, state=state, image=image, step=step,
        noise_seed=c.call_seed(row['task_name'], seed, decision))


def selector():
    path = c.BACKEND / 'gr00t/eval/stride_selector.py'
    spec = importlib.util.spec_from_file_location('authors_gr00t_selector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_exec_k_stride


def episode(conn, choose, row, index, seed, arm, phase, limit):
    import gymnasium as gym
    from robocasa.utils.gym_utils import gymnasium_groot  # registers GR1 tasks

    name = f't{index:02d}-s{seed:05d}-{arm}'
    output = c.RUN / phase / (name + '.json')
    if output.exists(): return json.loads(output.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_suffix('.trace.jsonl')
    if trace_path.exists():
        raise RuntimeError(f'Incomplete trace exists: {trace_path}; inspect before resuming')
    frozen = c.verify_freeze() if phase == 'test' else None
    rpc(conn, 'new_episode', arm=arm)
    random.seed(seed); np.random.seed(seed)
    started = time.perf_counter()
    record = dict(phase=phase, arm=arm, seed=seed, task_index=index, **row,
        status='RUNNING', gpu=c.CONFIG['gpu'], frozen_sha256=frozen,
        max_environment_steps=limit, policy_call_cap=(limit + 3) // 4,
        started_unix=time.time(), environment_steps=0, policy_calls=0)
    c.atomic(c.RUN / 'current.json', record)
    env = gym.make(row['env_name'], enable_render=True)
    try:
        native = env.unwrapped.env
        native.rng = np.random.default_rng(seed)
        obs, reset_info = env.reset(seed=seed)
        request = policy_request(obs, row, seed, 0, 0)
        record['initial_fingerprint'] = dict(state=c.array_hash(request['state']),
            image=c.array_hash(request['image']),
            physics=c.array_hash(native.sim.get_state().flatten()),
            instruction=obs['annotation.human.coarse_action'])
        record['simulator_declared_horizon'] = getattr(native, 'horizon', None)
        record['simulator_ignore_done'] = getattr(native, 'ignore_done', None)
        if phase == 'compatibility' and not (c.RUN / 'POLICY_PREFLIGHT.json').exists():
            c.atomic(c.RUN / 'POLICY_PREFLIGHT.json', rpc(conn, 'preflight', **request))
        posterior = {}; ts_rng = np.random.default_rng(0); history = []
        steps = calls = 0
        inference_s = simulator_s = selector_s = rpc_s = 0.
        tokens = reused = refreshes = 0
        token_counts_available = True
        ok = success(reset_info); first_success = 0 if ok else None
        terminated = truncated = False
        with trace_path.open('x') as stream:
            while steps < limit and not (terminated or truncated):
                if calls >= record['policy_call_cap']: raise RuntimeError('Policy-call ceiling reached')
                request = policy_request(obs, row, seed, calls, steps)
                start = time.perf_counter()
                pred = rpc(conn, 'infer', **request)
                rpc_s += time.perf_counter() - start
                inference_s += pred['policy_seconds']
                if pred['model_input_tokens'] is None: token_counts_available = False
                else: tokens += pred['model_input_tokens']
                actions = pred['actions']
                start = time.perf_counter()
                k, diagnostics = choose(pred['velocity'], 16, xt_step=actions,
                    history_actions=history, ts_state=posterior, ts_rng=ts_rng, **c.AHS)
                selector_s += time.perf_counter() - start
                k = int(k)
                if not 4 <= k <= 16: raise RuntimeError('Invalid authors AHS prefix')
                before = steps; actual = 0
                for action in actions[:k]:
                    split = np.split(action, np.cumsum(c.DIMS)[:-1])
                    start = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(
                        {'action.' + p: a for p, a in zip(c.PARTS, split)})
                    simulator_s += time.perf_counter() - start
                    steps += 1; actual += 1
                    history.append(np.asarray(action, dtype=np.float32).copy())
                    hit = success(info)
                    if hit and first_success is None: first_success = steps
                    ok = ok or hit
                    if steps >= limit or terminated or truncated: break
                rpc(conn, 'commit', executed=actual)
                reused += int(pred['sampling']['reused_positions'] > 0)
                refreshes += int(pred['sampling']['refresh'])
                line = dict(decision=calls, start_step=before, selected_k=k, executed=actual,
                    input_state_hash=c.array_hash(request['state']), input_image_hash=c.array_hash(request['image']),
                    actions_hash=pred['actions_hash'], fresh_noise_hash=pred['fresh_noise_hash'],
                    used_noise_hash=pred['used_noise_hash'], sampling=pred['sampling'], ahs=diagnostics,
                    policy_seconds=pred['policy_seconds'], model_input_tokens=pred['model_input_tokens'])
                if calls == 0: record['first_prediction'] = {key: line[key] for key in
                    ('actions_hash', 'fresh_noise_hash', 'used_noise_hash', 'selected_k')}
                stream.write(json.dumps(c.jsonable(line), sort_keys=True, allow_nan=False) + '\n'); stream.flush()
                calls += 1
                c.atomic(c.RUN / 'current.json', dict(record, environment_steps=steps,
                    policy_calls=calls, success_observed=ok, last_update_unix=time.time()))
        record.update(status='COMPLETED', success=bool(ok), first_success_step=first_success,
            environment_steps=steps, policy_calls=calls, inference_seconds=inference_s,
            policy_roundtrip_seconds=rpc_s, simulator_seconds=simulator_s, selector_seconds=selector_s,
            wall_seconds=time.perf_counter()-started, noise_reused_decisions=reused, noise_refreshes=refreshes,
            model_input_tokens=tokens if token_counts_available else None,
            token_metric='Sum of Eagle attention masks, including image tokens; no generated text',
            external_vlm_calls=0, paid_api_calls=0, horizon_violations=int(steps > limit),
            policy_budget_violations=int(calls > record['policy_call_cap']),
            terminated=bool(terminated), truncated=bool(truncated))
        c.atomic(output, record); c.atomic(c.RUN / 'current.json', record)
        print(json.dumps({k:record[k] for k in ('phase','arm','task_name','seed','success','environment_steps','policy_calls')}), flush=True)
        return record
    finally:
        env.close()


def paired_check(a, b):
    if a['initial_fingerprint'] != b['initial_fingerprint']:
        raise RuntimeError('Paired initial observations or simulator state differ')
    if a['first_prediction'] != b['first_prediction']:
        raise RuntimeError('Paired first prediction differs')


def report():
    from report import summarize
    r = summarize(c.RUN)
    c.atomic(c.RUN / 'SUMMARY.json', r)
    return r


def freeze():
    existing = c.RUN / 'FROZEN.json'
    if existing.exists(): c.verify_freeze(); return
    weights = {str(p): c.sha(p) for p in sorted(c.MODEL.glob('*.safetensors'))}
    weights.update({str(p):c.sha(p) for p in [c.MODEL/'config.json', c.MODEL/'experiment_cfg/metadata.json']})
    c.atomic(existing, dict(config=c.CONFIG, ahs=c.AHS,
        sources={str(p):c.sha(p) for p in c.source_paths()}, weights=weights,
        freeze_unix=time.time(), design='V4 adaptive copied unchanged; no GR00T tuning',
        v4_controller_sha256=c.sha(c.HERE/'controller_v4.py'),
        compatibility=c.sha(c.RUN/'POLICY_PREFLIGHT.json')))


def main():
    c.RUN.mkdir(parents=True, exist_ok=True)
    lock = (c.RUN / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    download = json.loads((c.HERE/'download.json').read_text())
    if download.get('status') != 'COMPLETE' or download['revision'] != c.REVISION:
        raise RuntimeError('Public checkpoint not completely downloaded at fixed revision')
    c.atomic(c.RUN/'PROTOCOL.json', dict(config=c.CONFIG, ahs=c.AHS, tasks=c.task_records()))
    if (c.RUN/'FROZEN.json').exists(): c.verify_freeze()
    else:
        with (c.RUN/'environment-sim.txt').open('w') as f:
            subprocess.run([sys.executable,'-m','pip','freeze','--all'],stdout=f,check=True)
    gpu = str(c.CONFIG['gpu'])
    snapshot = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free','--format=csv,noheader'], text=True)
    c.atomic(c.RUN/'hardware-start.json', dict(snapshot=snapshot, selected_gpu=int(gpu), time=time.time()))
    line = next(x for x in snapshot.splitlines() if x.split(',')[0].strip()==gpu)
    if int(line.split(',')[-1].strip().split()[0]) < 20000:
        raise RuntimeError('Selected GPU has less than 20 GB free; no other jobs will be stopped')
    key = c.RUN/'socket.auth'
    if not key.exists():
        fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:f.write(os.urandom(32))
    ready = c.RUN/'policy-ready.json'; ready.unlink(missing_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED='0')
    log=(c.RUN/'policy-worker.log').open('a')
    worker=subprocess.Popen(['/home/hitarth/gr00t-robocasa/.venv/bin/python','-u',str(c.HERE/'policy_worker.py')],
        env=env,stdout=log,stderr=subprocess.STDOUT)
    conn=None
    try:
        deadline=time.monotonic()+600
        while not ready.exists():
            if worker.poll() is not None:raise RuntimeError('Policy worker exited; inspect policy-worker.log')
            if time.monotonic()>deadline:raise RuntimeError('Policy loading exceeded ten minutes')
            time.sleep(1)
        conn=Client(str(c.RUN/'policy.sock'),family='AF_UNIX',authkey=key.read_bytes())
        choose=selector(); rows=c.task_records()
        smoke=[]
        for arm in c.CONFIG['arms']:
            smoke.append(episode(conn,choose,rows[0],0,902,arm,'compatibility',64))
        paired_check(*smoke)
        if not (c.RUN/'COMPATIBILITY.json').exists():
            c.atomic(c.RUN/'COMPATIBILITY.json',dict(status='PASS',native_equivalence=True,
                paired_initial_state_and_prediction=True,smoke_episodes=2,smoke_steps_per_arm=64,
                no_eval_tuning=True))
        freeze()
        plan=[]
        # Seed-major interleaving covers all tasks before the next repetition.
        for si,seed in enumerate(c.CONFIG['seeds']):
            for index,row in enumerate(rows):
                arms=c.CONFIG['arms'] if (si*24+index)%2==0 else list(reversed(c.CONFIG['arms']))
                plan.append(dict(task_index=index,seed=seed,arms=arms))
        c.atomic(c.RUN/'test-cases.json',plan)
        for case in plan:
            results={}
            for arm in case['arms']:
                summary=report()
                c.atomic(c.RUN/'status.json',dict(state='running',supervisor_pid=os.getpid(),
                    current_case=case,current_arm=arm,updated_unix=time.time(),**summary))
                results[arm]=episode(conn,choose,rows[case['task_index']],case['task_index'],case['seed'],arm,'test',720)
            paired_check(results['authors'],results['v4'])
            report()
        result=report()
        c.atomic(c.RUN/'status.json',dict(state='completed',supervisor_pid=os.getpid(),updated_unix=time.time(),**result))
        print(json.dumps(result),flush=True)
    finally:
        if conn is not None:
            try:conn.send(dict(op='shutdown'));conn.recv();conn.close()
            except Exception:pass
        if worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=20)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
        log.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        c.atomic(c.RUN/'LAST_ERROR.json',dict(time=time.time(),error=traceback.format_exc()))
        c.atomic(c.RUN/'status.json',dict(state='FAILED',updated_unix=time.time(),error=traceback.format_exc()))
        raise
