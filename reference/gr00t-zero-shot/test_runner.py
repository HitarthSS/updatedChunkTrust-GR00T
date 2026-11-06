"""Protocol invariants and matched-coverage accounting; no model download."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

import common as c
from controller_v4 import PROFILES, ValidatedNoise
from report import summarize


class RunnerTests(unittest.TestCase):
    def test_noise_feedback_and_prefix_shift(self):
        s=ValidatedNoise(np.ones(29),PROFILES['adaptive'])
        image=np.zeros((256,256,3),dtype=np.uint8)
        state=np.zeros(29); fresh=np.ones((16,32),dtype=np.float32)
        used,info=s.prepare(fresh,state,image,0)
        self.assertEqual(info['correlation'],0)
        actions=np.full((16,29),.02,dtype=np.float32)
        s.commit(used,state,image,actions,8,info)
        zero=np.zeros_like(fresh)
        used,info=s.prepare(zero,actions[7],image,8)
        self.assertEqual(info['reused_positions'],8)
        self.assertAlmostEqual(info['correlation'],.65)
        np.testing.assert_allclose(used[:8],.65)
        np.testing.assert_array_equal(used[8:],0)
        s.commit(used,actions[7],image,actions,8,info)
        used,info=s.prepare(fresh,state,np.full_like(image,255),16)
        self.assertEqual(info['correlation'],0)
        self.assertIn('large_scene_change',info['refresh_reasons'])

    def test_seed_and_budget(self):
        self.assertEqual(c.CONFIG['seeds'],[20,21,22,23,24])
        self.assertEqual(c.CONFIG['max_environment_steps']//min(c.AHS['stride_candidates']),180)
        self.assertEqual(c.call_seed('task',20,0),c.call_seed('task',20,0))
        self.assertNotEqual(c.call_seed('task',20,0),c.call_seed('task',20,1))

    def test_report_does_not_include_unmatched_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test';p.mkdir()
            for arm,seed in [('authors',20),('v4',20),('v4',21)]:
                r=dict(status='COMPLETED',arm=arm,task_index=0,seed=seed,success=arm=='v4',
                    initial_fingerprint='same',first_prediction='same',model_input_tokens=100)
                for key in ('policy_calls','inference_seconds','policy_roundtrip_seconds','simulator_seconds',
                    'selector_seconds','wall_seconds','environment_steps','noise_reused_decisions',
                    'horizon_violations','policy_budget_violations','paid_api_calls','external_vlm_calls'):r[key]=0
                (p/f't00-s{seed:05d}-{arm}.json').write_text(json.dumps(r))
            result=summarize(tmp)
            self.assertEqual(result['completed_episodes'],3)
            self.assertEqual(result['completed_cases'],1)
            self.assertEqual(result['arms']['v4']['successes'],1)
            self.assertEqual(result['paired']['helped'],1)


if __name__=='__main__':unittest.main()
