# Input sources and redistribution

## 141-bus network

Obtain `case141.m` from the MATPOWER distribution cited in the manuscript. The recorded computational environment uses MATPOWER 8.1. The case file itself is not included here. Verify it against the input hashes in the controlled-study records before a numerical reproduction.

- MATPOWER project: https://matpower.org/
- Official source repository: https://github.com/MATPOWER/matpower
- Store the acquired file at `reproduction/data/raw/case141.m` for the original controlled runner.

The supplied synthetic rating overlay belongs to this experiment. It is not a native equipment-rating dataset. The policy and calibration records are in `reproduction/outputs/rate_v4_design/`.

## Ausgrid Solar Home Electricity Data

Source: https://data.gov.au/data/dataset/nsw-solar-home-electricty-data

Use the three original half-hour CSV files covering July 2010–June 2013. File names, versions, byte hashes and preprocessing metadata are recorded under `reproduction/outputs/ees_transition/archive/091_data/processed/`. Raw household measurements and the derived full profile arrays are not redistributed in this deposit. Obtain them from the original source and follow its terms.

`reproduction/tools/prepare_ees_australian_case_inputs.py` contains the cleaning, time alignment, completeness rules and deterministic profile assignment. Its year labels `TRAIN`, `DEVELOPMENT` and `EVALUATION` are implementation identifiers for the temporal roles documented in the SI, not an assertion that a machine-learning model was used.

## CSIRO/GridQube network

- Collection DOI: https://doi.org/10.25919/ghnz-bk28
- Collection page: https://data.csiro.au/collection/csiro:65408

The collection's official metadata identifies **CC BY-NC-SA 4.0** and retains the notice “All Rights (including copyright) GridQube 2025.” A copyright notice does not cancel the stated licence. The two stored metadata records preserve both fields. The OpenDSS network files themselves are not copied into this repository; obtain them from the collection under its terms. The project's MIT and CC BY grants do not replace the collection's non-commercial and share-alike conditions where those conditions apply.

The representative LV model source and version identifiers are retained in their scenario protocols and metadata. The same source-access rule applies to these network files. Any later decision to redistribute inputs must preserve their source licences and notices.

## What is included

The deposit contains calculated interval/day results, figure tables, preprocessing and simulation code, scenario settings, selected-date lists, deterministic mapping tables, and source identifiers/hashes. It does not contain private utility operational records. Delivery quantities in the Australian experiment are model outputs derived from the cited profiles and network representation.

Third-party source DOIs above remain valid source citations. They are distinct from a DOI for this repository; no repository DOI is claimed.

See [the third-party notices](../THIRD_PARTY_NOTICES.md) and [licensing scope](../LICENSING.md). Author-generated numerical outputs are licensed only to the extent of the contributors' rights; upstream data, metadata and dependencies retain their own terms.
