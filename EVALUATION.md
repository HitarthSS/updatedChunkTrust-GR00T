# Evaluation: My harness for ChunkTrust — GR00T N1.5

Twenty-four RoboCasa GR1 tasks, five seeds each (20–24). Different policy from the pi0.5 repository.

## Budget

Our pilot: 720 physical actions and 180 policy calls. Released GR00T CLI defaults to 1,440; the paper's exact GR00T launch horizon is unverified.

## Interpretation

- This candidate performed worse overall: nine helped, sixteen harmed.

- The public checkpoint revision is pinned, but the paper did not supply an immutable checkpoint revision.

- Zero-shot benchmark deployment does not prove absence from all pretraining.

- This is a strict-budget matched pilot, not an exact paper replication or a pi0.5 result.

## Reproduction discipline

Use the reported condition only. Establish native-path parity, prepare upstream models/assets and declare case manifests before expensive runs. This export itself is a source publication, not a new evaluation. Keep all historical and new cohorts separate.
