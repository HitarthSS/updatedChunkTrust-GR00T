"""Isolated GR00T N1.5 pilot; no API credentials or paid services."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / 'workspaces/gr00t_n15_chunktrust'
BENCHMARK = ROOT / 'workspaces/robocasa-gr1-tabletop-tasks'
RUN = ROOT / 'comparison-gr00t-n15-v4'
REVISION = '869830fc749c35f34771aa5209f923ac57e4564e'
MODEL = ROOT / 'models/GR00T-N1.5-3B' / REVISION
PARTS = ('left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist')
DIMS = (7, 7, 6, 6, 3)
IMAGE = 'video.ego_view_bg_crop_pad_res256_freq20'
CONFIG = dict(
    policy='nvidia/GR00T-N1.5-3B', revision=REVISION,
    upstream_commit='4af2b622892f7dcb5aae5a3fb70bcb02dc217b96',
    chunktrust_commit='a9fd57412b3a19a0728cd726b0558dd89395fb84',
    embodiment='gr1', data_config='fourier_gr1_arms_waist',
    denoising_steps=4, generated_horizon=16, model_action_dim=32,
    max_environment_steps=720, policy_call_cap=180, pilot_budget_multiplier=1.0,
    paper_horizon_multiplier=None,
    seeds=[20, 21, 22, 23, 24], task_count=24, episodes_per_arm=120,
    arms=['authors', 'v4'], selected_profile='adaptive', gpu=6,
    training=False, qha=False, paid_api_calls=0,
    paper_ahs_percent=44.92, paper_bare_policy_percent=43.25,
    paper_episode_count='not explicitly stated for GR00T; do not infer from pi0.5',
    paper_checkpoint_revision='not supplied; fixed public revision recorded above',
    horizon_gap='GR00T released CLI defaults to 1440; pilot uses a stricter 720-step budget, matching our prior pi0.5 comparison. Exact paper GR00T launch command and episode limit unavailable. No claim that 720 is the paper GR00T horizon.',
    interpretation='24-task matched pilot, not a full paper reproduction. Zero-shot means no benchmark post-training; absence from pretraining unverified.',
    timing='One shared GPU, serial interleaved arms; descriptive wall-clock timing only.',
)
AHS = dict(
    stride_candidates=[4, 8, 12, 16], exec_mode='expected_round',
    expected_temp=1.0, stride_alpha=4.0, intra_cut_t=.25,
    z_mode='window_rms', tau_window=3, signal_mode='both',
    use_speed_uniformity=True, speed_window=20, speed_weight=.5,
    speed_beta=4., speed_fallback='ehh_only', ts_epsilon=.05,
    ts_update_mode='kernel_forget', ts_kernel_bandwidth=10.,
    ts_forget_rho=.99, ts_update_eta=1.,
)


def configure():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', WANDB_MODE='disabled',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def jsonable(v):
    import math
    import numpy as np
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return jsonable(v.tolist())
    if isinstance(v, np.generic): return jsonable(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def array_hash(value):
    import numpy as np
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def source_paths():
    paths = list(HERE.glob('*.py')) + [HERE / 'README.md', BACKEND / '.chunktrust-install.json']
    paths += sorted((BACKEND / 'gr00t').rglob('*.py'))
    paths += sorted((BENCHMARK / 'robocasa/utils/gym_utils').glob('*.py'))
    paths += [ROOT / 'models/pi05_robocasa_gr1_tabletop24_step30000/configs/robocasa_gr1_tabletop24_manifest.json']
    return paths


def verify_freeze():
    value = json.loads((RUN / 'FROZEN.json').read_text())
    for path, expected in value['sources'].items():
        if sha(path) != expected: raise RuntimeError('Frozen source changed: ' + path)
    if value['config'] != CONFIG or value['ahs'] != AHS:
        raise RuntimeError('Frozen configuration changed')
    return sha(RUN / 'FROZEN.json')


def task_records():
    import re
    path = ROOT / 'models/pi05_robocasa_gr1_tabletop24_step30000/configs/robocasa_gr1_tabletop24_manifest.json'
    rows = []
    for item in json.loads(path.read_text())['tasks']:
        repo = item['repo_id']
        match = re.fullmatch(r'gr1_unified\.(.+)_GR1ArmsAndWaistFourierHands_1000', repo)
        if not match: raise ValueError(repo)
        task = match[1]
        rows.append(dict(task_name=task, repo_id=repo,
            env_name=f'gr1_unified/{task}_GR1ArmsAndWaistFourierHands_Env'))
    if len(rows) != 24 or len({r['env_name'] for r in rows}) != 24: raise RuntimeError('Wrong task scope')
    return rows


def call_seed(task, seed, decision):
    material = f'chunktrust-gr00t-n15-v4-pilot|{task}|{seed}|{decision}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'big') % (2**63 - 1)
