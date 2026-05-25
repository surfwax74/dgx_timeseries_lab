"""Top-level CLI dispatcher.

Splits ``dgx-ts <subcommand> [hydra args]`` so each subcommand can use its
own Hydra entrypoint without conflict. Phase 0 only ships ``train``; other
subcommands print a friendly "not yet implemented" message that points at
the phase that will introduce them.
"""

from __future__ import annotations

import sys

_USAGE = """dgx-ts <subcommand> [hydra-style overrides]

Subcommands:
  train      Fit a detector against a dataset (Phase 0+)
  eval       Run a detector against a dataset, report metrics (Phase 2)
  benchmark  Run the bake-off across multiple detectors (Phase 2)
  synth      Generate synthetic telemetry datasets to disk (Phase 1)
  export     Emit ONNX + model_card + feature_schema for MLOps lift (Phase 5)

Example:
  dgx-ts train experiment=phase0_smoke
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_USAGE)
        raise SystemExit(0 if len(sys.argv) >= 2 else 1)

    subcommand = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]  # strip subcommand for Hydra

    if subcommand == "train":
        from .train import run

        run()
        return

    if subcommand == "synth":
        from .synth import run

        run()
        return

    if subcommand == "benchmark":
        from .benchmark import run

        run()
        return

    if subcommand == "export":
        from .export import run

        run()
        return

    if subcommand == "eval":
        print("dgx-ts eval: arrives in Phase 2.5 (use `dgx-ts benchmark` for now).")
        raise SystemExit(2)

    print(f"dgx-ts: unknown subcommand '{subcommand}'\n\n{_USAGE}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
