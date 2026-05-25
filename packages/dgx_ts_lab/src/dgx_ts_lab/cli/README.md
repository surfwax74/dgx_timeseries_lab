# dgx_ts_lab.cli

The `dgx-ts` command-line interface. Subcommand dispatcher + per-subcommand entrypoints, each Hydra-wired.

## Files

| File | Subcommand | Phase |
|---|---|---|
| `main.py` | dispatcher — splits `dgx-ts <subcommand> [hydra args]` so each subcommand gets its own Hydra entry. | 0 ✓ |
| `train.py` | `dgx-ts train experiment=…` — fits a detector, logs to MLflow. | 0 ✓ |
| `synth.py` | `dgx-ts synth dataset=… output_dir=…` — materializes a dataset to parquet. | 1 ✓ |
| `eval.py` | `dgx-ts eval` — score a pre-trained detector. | 2 (planned) |
| `benchmark.py` | `dgx-ts benchmark` — bake-off across detectors × datasets. | 2 (planned) |
| `export.py` | `dgx-ts export` — emit ONNX + model_card + feature_schema. | 5 (planned) |

## How a subcommand wires Hydra

The dispatcher pops `sys.argv[1]` (the subcommand name) before handing control to the per-subcommand function. That function is decorated with `@hydra.main(...)` and sees only its own args — no conflicts between subcommands.

`config_path` is resolved from `__file__` at import time so the CLI works whether invoked from the repo root, from `outputs/.../`, or as an installed entry point.

## Convention for adding a subcommand

1. Write `cli/my_cmd.py` with a `@hydra.main(...)`-decorated `run()` function.
2. Add a branch in `cli/main.py` (also update `_USAGE`).
3. Add `configs/experiment/my_cmd_example.yaml` showing typical invocation.
4. Document it here.
