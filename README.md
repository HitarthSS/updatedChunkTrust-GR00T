# My harness for ChunkTrust — GR00T N1.5

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2609.39754). The evaluated behavior is in [`my_harness.py`](my_harness.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | 44.92% AHS for GR00T N1.5 |

| Our run of the authors' harness | 55/120 (45.83%) |

| **My harness** | **48/120 (40.00%)** |


Twenty-four RoboCasa GR1 tasks, five seeds each (20–24). Different policy from the pi0.5 repository.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Apply prediction-validated, time-aligned noise reuse to the frozen GR00T N1.5 policy.

2. Refresh noise when tracking or visual evidence changes, while retaining the released AHS selector.

3. Keep policy weights, four denoising steps, native action coordinates and one sample per decision unchanged.


Categories: `sampler_guidance`, `execution_monitoring`.

## Models and inputs

- nvidia/GR00T-N1.5-3B at 869830fc749c35f34771aa5209f923ac57e4564e, GR1 embodiment.

- The GR00T backend is pinned to 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 in the archived configuration.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/hf618/ChunkTrust) before installing its model stack.
2. Controller dependency: numpy==1.26.4
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 -m pip install "numpy==1.26.4"
python3 tools/verify_export.py
python -m unittest discover -s tests -v
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

Use ValidatedNoise.prepare/commit around the native GR00T sampling call exactly as for the pi0.5 controller. reference/gr00t-zero-shot contains the evaluated backend mapping and action-loop integration.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

Our pilot: 720 physical actions and 180 policy calls. Released GR00T CLI defaults to 1,440; the paper's exact GR00T launch horizon is unverified.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `my_harness.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- This candidate performed worse overall: nine helped, sixteen harmed.

- The public checkpoint revision is pinned, but the paper did not supply an immutable checkpoint revision.

- Zero-shot benchmark deployment does not prove absence from all pretraining.

- This is a strict-budget matched pilot, not an exact paper replication or a pi0.5 result.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

3 CPU tests passed. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
