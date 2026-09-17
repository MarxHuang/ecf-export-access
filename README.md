# Delivery-informed allocation of electricity exports

Research materials for **Increasing electricity exports through delivery-informed allocation in distribution networks**.

The study examines when unused export permission prevents other participants from delivering electricity. Externality-conditioned counterfactual feedback (ECF) checks a completed operating interval and adjusts later request ceilings only when the comparison identifies recoverable delivery. The allocation rule and network limits remain unchanged.

## Start here

- [Manuscript](paper/main.pdf) and [Supporting Information](paper/supporting_information.pdf).
- [Figure data and figure-number mapping](docs/FIGURE_DATA.md), including Excel and CSV files.
- [Reproduction instructions](docs/REPRODUCING.md), with a quick check that requires only Python's standard library.
- [Input sources and redistribution status](docs/DATA_SOURCES.md).
- [Version and experiment map](docs/EXPERIMENTS.md).

This repository is a **private author-review deposit** as of 17 September 2026. A public release and reuse licence have not yet been approved. The repository URL is real; no repository DOI has been assigned. The manuscript has not been represented here as an accepted publication.

## Contents

| Directory | Contents |
|---|---|
| `paper/` | Current main text, SI, TeX dependencies, and the figures used in those documents |
| `figure_data/` | Source tables for the current main and supplementary figures, plus tabulated sensitivities |
| `reproduction/src/` and `reproduction/tools/` | Controlled-study implementation and preprocessing/analysis tools |
| `reproduction/outputs/` | Selected parameter files, calculated results, and the frozen Australian implementation |
| `provenance/` | File hashes and source-file correspondence |
| `tools/` | Deposit verification, lossless expansion, and portable entry points |

The historical directory and variable names in the numerical source are retained so that saved hashes and results can be traced. They are implementation identifiers, not additional manuscript versions. In particular, the Australian results here use the current loading and voltage-margin settings; the superseded 4.564/5.220 MWh comparison is not the primary result of this deposit.

## Quick verification

From the repository root:

```console
python tools/verify_deposit.py
python tools/run_public_case.py --list
python tools/run_public_case.py --case primary
```

The first command checks the deposited source hashes, reads all 30 Australian case tables, and recalculates the primary energy comparisons. The last command is a **dry run**: it reports which externally acquired inputs are still needed. It does not start a simulation without `--execute`.

The primary ECF-minus-no-feedback comparison is approximately **+1.392 MWh delivered export** over 80 assessed days. Total authorization rises by about **1.206 MWh**, while unused authorization falls by about **0.186 MWh**. These signs follow the current results and are not replaced with the earlier interpretation that total authorization always falls.

## Source data and reuse

Ausgrid measurements and the CSIRO/GridQube networks must be obtained from their original distributors under their applicable terms. This deposit provides source identifiers, processing code, mappings, date selections, and calculated results; it does not redistribute their raw measurement files or network models. See [DATA_SOURCES.md](docs/DATA_SOURCES.md).

No general open-source or open-data licence is granted by this private deposit. Original third-party notices continue to apply. Before public release, the authors should approve a licence and confirm the distribution rights for each included material. Author-review and journal-access arrangements are separate from permission to reuse materials.
