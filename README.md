# Delivery-informed allocation of electricity exports

Research materials for **Increasing electricity exports through delivery-informed allocation in distribution networks**.

The study examines when unused export permission prevents other participants from delivering electricity. Externality-conditioned counterfactual feedback (ECF) checks a completed operating interval and adjusts later request ceilings only when the comparison identifies recoverable delivery. The allocation rule and network limits remain unchanged.

## Start here

- [Manuscript](paper/main.pdf) and [Supporting Information](paper/supporting_information.pdf).
- [Figure data and figure-number mapping](docs/FIGURE_DATA.md), including Excel and CSV files.
- [Reproduction instructions](docs/REPRODUCING.md), with a quick check that requires only Python's standard library.
- [Input sources and redistribution status](docs/DATA_SOURCES.md).
- [Version and experiment map](docs/EXPERIMENTS.md).
- [File-by-file licensing](LICENSING.md) and [repository status](docs/PUBLIC_RELEASE.md).

This repository is **public** as of 24 September 2026. It contains author-generated code, calculated results, source-data tables and compiled author manuscript PDFs. It has no repository DOI, and the paper is not an accepted publication. Code/data licences and manuscript-PDF rights differ as described below.

The 24 September revision updates the manuscript and SI for a prospective Nature Communications submission. It adds a frequency-calibrated replay-free control to distinguish replay-based update selection from simply updating less often. The expanded comparison reports all 12 fixed schedules: the original three are retained unchanged, and nine additional schedules are deposited with daily and interval results. The diagnostic was designed after the original 80-day result; it is not an independent validation. Manuscript TeX, bibliography/style files, standalone manuscript figures and publisher compilation assets remain absent from the current tree. Older public commits and tags may still expose those withdrawn sources; no history rewrite has been performed.

## Contents

| Directory | Contents |
|---|---|
| `paper/` | Compiled author manuscript and SI PDFs only, with a sharing notice; no manuscript compilation sources |
| `figure_data/` | Source tables for the current main and supplementary figures, plus tabulated sensitivities |
| `reproduction/src/` and `reproduction/tools/` | Controlled-study implementation and preprocessing/analysis tools |
| `reproduction/outputs/` | Selected parameter files, calculated results, and the frozen Australian implementation |
| `reproduction/diagnostics/qualification_value/` and `qualification_value_12seed/` | Original and expanded post hoc replay-selection designs and exact local analysis code |
| `provenance/` | File hashes and source-file correspondence |
| `tools/` | Deposit verification, lossless expansion, and portable entry points |

The historical directory and variable names in the numerical source are retained so that saved hashes and results can be traced. They are implementation identifiers, not additional manuscript versions. In particular, the Australian results here use the current loading and voltage-margin settings; the superseded 4.564/5.220 MWh comparison is not the primary result of this deposit.

## Quick verification

From the repository root:

```console
python tools/verify_deposit.py
python tools/verify_qualification_value.py
python tools/verify_qualification_value_12seed.py
python tools/run_public_case.py --list
python tools/run_public_case.py --case primary
```

The first command checks deposited source hashes and primary energy arithmetic. The next two check the retained three-schedule and expanded 12-schedule diagnostics, respectively. The latter verifies 960 daily records and 46,080 interval records, including numerical agreement of the first three schedules with the earlier deposit. The last command is a **dry run**: it reports externally acquired inputs still needed and does not start a simulation without `--execute`.

The primary ECF-minus-no-feedback comparison is approximately **+1.392 MWh delivered export** over 80 assessed days. Total authorization rises by about **1.206 MWh**, while unused authorization falls by about **0.186 MWh**. These signs follow the current results and are not replaced with the earlier interpretation that total authorization always falls.

With ECF's update formula but no replay test, updating after every unused award reduces delivery by 6.423 MWh relative to ECF. That comparison also changes update frequency. Across 12 replay-free schedules calibrated to a similar expected frequency, ECF delivers 0.917–1.596 MWh more, averaging 1.343 MWh. The schedule mean is only 0.049 MWh above no feedback. These are modeled 80-day differences, not field-metered gains.

## Source data and reuse

Ausgrid measurements and the CSIRO/GridQube networks must be obtained from their original distributors under their applicable terms. This deposit provides source identifiers, processing code, mappings, date selections, and calculated results; it does not redistribute their raw measurement files or network models. See [DATA_SOURCES.md](docs/DATA_SOURCES.md).

| Material | Applied terms |
|---|---|
| Original simulation, analysis and reproduction code, tests, and software configuration | [MIT](LICENSES/MIT.txt) |
| Author-owned calculated results, figure data and explanatory documentation | [CC BY 4.0](LICENSES/CC-BY-4.0.txt) |
| Main manuscript and SI PDFs | All rights reserved; no open article licence is granted by this repository |
| Third-party metadata, input materials and dependencies | Their own terms; not relicensed by this project |

See [LICENSING.md](LICENSING.md) for exact scope, [the per-file register](provenance/LICENSE_MANIFEST.csv), and [third-party notices](THIRD_PARTY_NOTICES.md). MIT and CC BY permit commercial reuse of material within their respective scopes. These grants cover only rights held by the contributors; they do not override upstream restrictions or grant rights to third-party source data, publisher artwork, logos or fonts. Code snippets in the documentation are MIT-licensed.

The repository is already public. Removing source files from the current tree did not remove older copies from Git history; see [the repository-status note](docs/PUBLIC_RELEASE.md). Do not treat the current branch's absence of manuscript source as proof that historical source is inaccessible.
