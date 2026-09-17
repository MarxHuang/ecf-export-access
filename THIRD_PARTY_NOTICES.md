# Third-party materials and dependencies

Project licences apply only to rights held by the contributors. Source access,
source citation and permission to redistribute are distinct.

## CSIRO / GridQube

Geth, Frederik; Heidari, Rahmat; Clark, Jordan; Lucas, Kurt; and Nimalsiri, Nanduni
(2025). *Realistic Australian Medium Voltage Feeder with Associated Low Voltage
Feeders*, v1. CSIRO Data Collection. https://doi.org/10.25919/ghnz-bk28.

The official collection metadata, checked on 17 September 2026, states
**Creative Commons Attribution Noncommercial-Share Alike 4.0**
(https://creativecommons.org/licenses/by-nc-sa/4.0/) and retains the notice
“All Rights (including copyright) GridQube 2025.”
These statements coexist: a copyright notice is not evidence that no licence exists.

The copied `CSIRO_COLLECTION_METADATA.json` and `SCHEMA_ORG_METADATA.json`
under `reproduction/outputs/ees_transition/archive/091_data/external/csiro_australian_mv_lv_feeder_v1/`
are source records, not original author prose. They remain unmodified and are
excluded from our MIT and CC BY grants. The network files are not redistributed;
obtain them from the collection and observe its attribution, non-commercial
and share-alike requirements as applicable. The collection licence text is
included in `LICENSES/CC-BY-NC-SA-4.0.txt` for reference.

## Ausgrid

The Solar Home Electricity Data come from
https://data.gov.au/data/dataset/nsw-solar-home-electricty-data.
Raw household measurements and the complete derived profile arrays are not
included. Obtain those inputs from their distributor and check the supplied
licence and conditions. Our project licence is not a licence for Ausgrid data.
Source filenames, hashes and processing records identify the inputs.

## MATPOWER and software dependencies

`case141.m` must be obtained from the MATPOWER distribution
(https://matpower.org/; https://github.com/MATPOWER/matpower).
It is not included here. The project's synthetic rating overlay is an original
experiment input, not the network's native equipment-rating data.

The required solver and analysis libraries are listed in
`reproduction/pyproject.toml` and the saved computational-environment record.
Their distributions are not vendored into this repository. Their own licences,
attributions and any commercial-solver conditions continue to apply.

## Manuscript assets

The current tree excludes manuscript TeX, the RSC bibliography style, standalone
manuscript figures and publisher header/footer assets. Compiled author PDFs may
still contain publisher template artwork or embedded fonts. No rights to those
components, logos or trademarks are granted by the project. Earlier Git history
retains the withdrawn files and their original rights.
