# Integration and arm definitions

Use the same `ValidatedNoise.prepare/commit` contract as pi0.5, with the GR1 action/state mapping in `reference/gr00t-zero-shot/policy_worker.py` and `test_runner.py`. Shapes are RGB `(256,256,3)`, state/ranges `(29,)`, noise `(16,32)` float32, and native actions `(16,29)`. Reset the controller each episode.

The authors' arm uses native fresh noise and released AHS. My harness changes only noise initialization using past public tracking evidence; it does not add a critic, sample or policy update. GR00T N1.5 uses four denoising steps. The 720-action pilot budget is explicitly different from the released CLI's 1,440 default; do not silently inherit the CLI default. The reference pipeline also retains a historical Python executable path that must be rebound to a prepared environment for a new, separately labeled evaluation.
