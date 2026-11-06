"""Persistent, local Unix-socket policy worker using the authors' backend."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
from multiprocessing.connection import Listener
from unittest.mock import patch

import common as c
c.configure()
sys.path.insert(0, str(c.HERE / 'deps'))
sys.path.insert(0, str(c.BACKEND))

import numpy as np
import torch
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from controller_v4 import PROFILES, ValidatedNoise


class Worker:
    def __init__(self):
        torch.manual_seed(0)
        data = load_data_config(c.CONFIG['data_config'])
        self.policy = Gr00tPolicy(model_path=str(c.MODEL), embodiment_tag='gr1',
            modality_config=data.modality_config(), modality_transform=data.transform(),
            denoising_steps=c.CONFIG['denoising_steps'])
        self.policy.model.requires_grad_(False)
        self.policy.model.eval()
        if any(p.requires_grad for p in self.policy.model.parameters()): raise RuntimeError('Unfrozen parameter')
        if self.policy.model.action_horizon != 16 or self.policy.model.action_dim != 32:
            raise RuntimeError('Unexpected model dimensions')
        self.native_get = self.policy._get_action_from_normalized_input
        self.tokens = None

        def measured(normalized_input, return_trace=False):
            mask = normalized_input.get('eagle_attention_mask')
            self.tokens = int(mask.sum().item()) if mask is not None else None
            return self.native_get(normalized_input, return_trace=return_trace)

        self.policy._get_action_from_normalized_input = measured
        stats = json.loads((c.MODEL / 'experiment_cfg/metadata.json').read_text())['gr1']['statistics']['action']
        self.ranges = np.concatenate([np.array(stats[p]['q99']) - np.array(stats[p]['q01']) for p in c.PARTS])
        self.sampler = None
        self.pending = None

    def episode(self, r):
        if self.pending is not None: raise RuntimeError('Uncommitted policy output')
        self.sampler = ValidatedNoise(self.ranges, PROFILES['adaptive']) if r['arm'] == 'v4' else None
        self.arm = r['arm']
        return dict(status='OK', ranges=self.ranges)

    def predict(self, r, instrument=True):
        if self.pending is not None: raise RuntimeError('Uncommitted policy output')
        torch.manual_seed(r['noise_seed'])
        observation = r['observation']
        original_randn = torch.randn
        captured = []

        def draw(*args, **kwargs):
            fresh = original_randn(*args, **kwargs)
            if tuple(fresh.shape) != (1, 16, 32) or captured:
                raise RuntimeError('Unexpected random draw in policy')
            noise = fresh[0].float().cpu().numpy().copy()
            if self.sampler is not None:
                used, info = self.sampler.prepare(noise, r['state'], r['image'], r['step'])
                supplied = torch.as_tensor(used, dtype=fresh.dtype, device=fresh.device).unsqueeze(0)
            else:
                supplied = fresh
                info = dict(reused_positions=0, correlation=0., refresh=True,
                    refresh_reasons=['independent_sampling'], next_low_progress_count=0)
            actual = supplied[0].float().cpu().numpy().copy()
            captured.append(dict(fresh=noise, used=actual, sampling=info))
            return supplied

        torch.cuda.synchronize()
        start = time.perf_counter()
        if instrument:
            with patch.object(torch, 'randn', draw):
                actions, info = self.policy.get_action(observation, options={'return_trace': True})
        else:
            actions, info = self.policy.get_action(observation, options={'return_trace': True})
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        flat = np.concatenate([np.asarray(actions['action.' + p], dtype=np.float32) for p in c.PARTS], axis=-1)
        velocity = np.asarray(info['denoise_trace']['v'], dtype=np.float32)
        if velocity.shape != (1, 4, 16, 32) or flat.shape != (16, 29):
            raise RuntimeError(f'Unexpected shapes: {flat.shape}, {velocity.shape}')
        if not np.isfinite(flat).all() or not np.isfinite(velocity).all(): raise RuntimeError('Non-finite prediction')
        response = dict(actions=flat, velocity=velocity[0], policy_seconds=elapsed,
            model_input_tokens=self.tokens, actions_hash=c.array_hash(flat))
        if instrument:
            if len(captured) != 1: raise RuntimeError('Missing policy noise draw')
            used = captured[0]
            response.update(sampling=used['sampling'], fresh_noise_hash=c.array_hash(used['fresh']),
                used_noise_hash=c.array_hash(used['used']))
            self.pending = dict(used=used['used'], state=r['state'], image=r['image'],
                actions=flat, sampling=used['sampling'])
        return response

    def preflight(self, r):
        if self.arm != 'authors': raise RuntimeError('Native equivalence test requires authors arm')
        a = self.predict(r, instrument=False)
        b = self.predict(r, instrument=True)
        self.pending = None
        if not np.array_equal(a['actions'], b['actions']) or not np.array_equal(a['velocity'], b['velocity']):
            raise RuntimeError('Instrumented author policy differs from native inference')
        return dict(status='PASS', exact_native_equivalence=True, policy_calls=2,
            actions_hash=a['actions_hash'], input_tokens=b['model_input_tokens'])

    def commit(self, r):
        if self.pending is None: raise RuntimeError('No prediction to commit')
        p = self.pending
        if self.sampler is not None:
            self.sampler.commit(p['used'], p['state'], p['image'], p['actions'], r['executed'], p['sampling'])
        self.pending = None
        return dict(status='OK')


def main():
    worker = Worker()
    socket = c.RUN / 'policy.sock'
    socket.unlink(missing_ok=True)
    key = (c.RUN / 'socket.auth').read_bytes()
    with Listener(str(socket), family='AF_UNIX', authkey=key) as listener:
        os.chmod(socket, 0o600)
        c.atomic(c.RUN / 'policy-ready.json', dict(pid=os.getpid(), model=str(c.MODEL),
            revision=c.REVISION, torch=torch.__version__, gpu=os.environ['CUDA_VISIBLE_DEVICES'],
            frozen=True, denoising_steps=4, ranges=worker.ranges))
        with listener.accept() as conn:
            while True:
                r = conn.recv()
                try:
                    op = r.pop('op')
                    if op == 'shutdown': conn.send(dict(status='OK')); break
                    result = getattr(worker, {'new_episode': 'episode', 'infer': 'predict'}.get(op, op))(r)
                    conn.send(dict(ok=True, result=result))
                except BaseException:
                    conn.send(dict(ok=False, error=traceback.format_exc()))
                    raise


if __name__ == '__main__': main()
