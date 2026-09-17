# Reproduction instructions

## 1. Verify the deposited files and reported arithmetic

Python 3.12 is recommended. This command only uses the standard library:

```console
python tools/verify_deposit.py
```

It checks every file in `provenance/SOURCE_MANIFEST.json`, verifies lossless gzip expansion, reconciles the Australian day tables with the run summaries, and compares the primary energy differences against Fig. 9 data. It does not rerun an AC power flow or establish that a physical model is correct.

## 2. Read or expand larger files

JSON/CSV files above 2 MB may be gzip-compressed. Python, R and many analysis tools can read them directly. For the legacy analysis programs that expect an uncompressed filename:

```console
python tools/expand_results.py
```

The command creates a separate `expanded/` tree and verifies every expanded file against its recorded original byte hash. It does not overwrite the deposited files.

## 3. Computational environment

The recorded environment is in `reproduction/outputs/ees_transition/archive/093_EES_COMPUTATIONAL_ENVIRONMENT.yaml`. It specifies Python 3.12.11, NumPy 2.3.0, pandas 2.3.0, SciPy 1.15.2, DSS-Python 0.15.7 with backend 0.14.5, CVXPY 1.9.1, OSQP 1.1.3 and other dependencies. The controlled-study package metadata is `reproduction/pyproject.toml`.

Use a separate environment. Do not replace the frozen Australian modules with the similarly named controlled/preprocessing modules. Differences in solver/backend versions should be recorded when comparing a new run with the deposited values.

## 4. Australian primary and existing sensitivity cases

```console
python tools/run_public_case.py --list
python tools/run_public_case.py --case primary
```

The second command resolves the original protocol paths relative to this checkout, prints the unchanged experiment arguments, and reports missing external inputs. It is a dry run. Obtain the external source files and construct the processed profile package described in `DATA_SOURCES.md` before proceeding. For preprocessing options:

```console
python reproduction/tools/prepare_ees_australian_case_inputs.py --help
```

After all inputs are present:

```console
python tools/run_public_case.py --case primary --execute --workers 2
```

New results go into `reproduction/reruns/primary/`. Existing results are never overwritten by this entry point. `--limit-days 1` requests a labelled smoke run, not a replacement for the full result. Other existing case IDs are listed by `--list`. The launcher modifies only path resolution, the worker count and explicitly requested smoke-run length; it does not choose gains, dates, load scales, ratings or other experimental parameters.

The archived extra-study scripts are also included. Some depend on auxiliary captured-ray or per-connection trajectory caches that are not part of this deposit, and contain original absolute-path checks. They require reconstruction of those caches and relocation before a full rerun. See `EXPERIMENTS.md`. Do not interpret the presence of their source as a claim that every historical script runs unchanged on a fresh computer.

## 5. Controlled study

From `reproduction/`, add `src` and the current directory to `PYTHONPATH`, then use module entry points:

```console
python -m tools.run_v16_externality_conditioned_feedback --help
python -m tools.analyze_v16_externality_conditioned_feedback --help
python -m tools.audit_v16_realized_delivery_ac --help
```

Pass explicit `--case`, `--policy`, `--source`, `--execution`, `--behavior`, `--design` and output paths. The compressed calibration and behavior files can be expanded with step 2. Use a new output directory and the deposited frozen design; do not call `--freeze-design` to replace it. The required network input must be obtained separately as described in `DATA_SOURCES.md`.

## 6. Read the paper

The `paper/` directory contains the compiled author manuscript and SI. Their TeX files, bibliography/style files, standalone figures and publisher compilation assets have been withdrawn from the current repository tree. Manuscript compilation is therefore not a supported operation in this checkout. This does not remove the numerical code, saved results or figure-data tables used to inspect and reproduce the scientific calculations.

## Verification status

`provenance/DEPOSIT_CHECK.json` records checks run when preparing this deposit. These checks cover file integrity, tabulated results, selected source unit tests and dry-run path resolution. They do not claim that the full simulation suite was rerun from newly downloaded third-party data during repository preparation.
