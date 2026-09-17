# Delivery-informed allocation of electricity exports

Research materials for **Increasing electricity exports through delivery-informed allocation in distribution networks**.

The study examines when unused export permission prevents other participants from delivering electricity. Externality-conditioned counterfactual feedback (ECF) checks a completed operating interval and adjusts later request ceilings only when the comparison identifies recoverable delivery. The allocation rule and network limits remain unchanged.

## Start here

- [Manuscript](paper/main.pdf) and [Supporting Information](paper/supporting_information.pdf).
- [Figure data and figure-number mapping](docs/FIGURE_DATA.md), including Excel and CSV files.
- [Reproduction instructions](docs/REPRODUCING.md), with a quick check that requires only Python's standard library.
- [Input sources and redistribution status](docs/DATA_SOURCES.md).
- [Version and experiment map](docs/EXPERIMENTS.md).
- [File-by-file licensing](LICENSING.md) and [public-release checklist](docs/PUBLIC_RELEASE.md).

This repository is a **private author-review deposit** as of 17 September 2026. The authors have authorized the scoped reuse licences below. This does not change repository visibility: editors and reviewers still need working access. The repository URL is unchanged; no repository DOI has been assigned. The manuscript is not presented as an accepted publication.

The `author-review-2026-09-17-r4` revision updates the main PDF to restore six omitted author email addresses from the earlier IEEE manuscript, retaining the two already present. All eight addresses have name labels and mailto links. Only the author-contact footnotes changed; the abstract, scientific text, SI, numerical code, result tables and figure data are unchanged. Manuscript TeX, bibliography/style files, standalone manuscript figures and publisher compilation assets remain absent from the current tree following r3. Older commits and tags still contain manuscript sources; this update does not erase Git history.

## Contents

| Directory | Contents |
|---|---|
| `paper/` | Compiled author manuscript and SI PDFs only, with a sharing notice; no manuscript compilation sources |
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

| Material | Applied terms |
|---|---|
| Original simulation, analysis and reproduction code, tests, and software configuration | [MIT](LICENSES/MIT.txt) |
| Author-owned calculated results, figure data and explanatory documentation | [CC BY 4.0](LICENSES/CC-BY-4.0.txt) |
| Main manuscript and SI PDFs | All rights reserved; no open article licence is granted by this repository |
| Third-party metadata, input materials and dependencies | Their own terms; not relicensed by this project |

See [LICENSING.md](LICENSING.md) for exact scope, [the per-file register](provenance/LICENSE_MANIFEST.csv), and [third-party notices](THIRD_PARTY_NOTICES.md). MIT and CC BY permit commercial reuse of material within their respective scopes. These grants cover only rights held by the contributors; they do not override upstream restrictions or grant rights to third-party source data, publisher artwork, logos or fonts. Code snippets in the documentation are MIT-licensed.

The repository remains private. In particular, **do not make the existing history public while manuscript-source disclosure is unwanted**. See [PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md) for the access and history checks still needed before release.
