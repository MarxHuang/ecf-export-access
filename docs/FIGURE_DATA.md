# Figure data

Use the current manuscript figure numbers rather than historical filenames.
The independent map below governs current figure-to-worksheet correspondence.
The eight workbooks are byte-identical to the preserved deposited files;
`provenance/RELEASE_CHECK_20261024_APEN_R3.json` records the exact hashes.
The `source_data/` names in the map are the local Supplementary data S1 names.
Their corresponding repository paths are:

| S1 workbook | Repository file |
|---|---|
| `bounded_record_feedback.xlsx` | `figure_data/bounded_record/Bounded_record_feedback.xlsx` |
| `controlled_figures.xlsx` | `figure_data/nc_submission/controlled_figures.xlsx` |
| `participant_adaptation.xlsx` | `figure_data/participant_adaptation/Participant_adaptation.xlsx` |
| `profile_mapping.xlsx` | `figure_data/profile_mapping/Profile_mapping.xlsx` |
| `public_replay_figures.xlsx` | `figure_data/australian_main/Fig9-11_Origin_data.xlsx` |
| `pv_capacity.xlsx` | `figure_data/pv_capacity/PV_capacity.xlsx` |
| `qualification_value.xlsx` | `figure_data/qualification_value/qualification_value.xlsx` |
| `request_classes_fig3c.xlsx` | `figure_data/controlled/Fig03c_request_choice_classes.xlsx` |

| Article item | Workbook in `source_data/` | Worksheets |
|---|---|---|
| Figures 1–2 | No numerical workbook | Explanatory diagrams; illustrative quantities are in their legends |
| Figure 3 | `controlled_figures.xlsx`; `request_classes_fig3c.xlsx` | `F02A_UtilityCurve`, `F02BC_ChoiceSurface`, `F02D_ConsequenceEq`; `Fig3c` |
| Figure 4 | `controlled_figures.xlsx` | `F06AB_Beta`, `F06C_PairedTargets`, `F06D_ReplayDist` |
| Figure 5 | `controlled_figures.xlsx` | `F07_Robustness` |
| Figure 6 | `public_replay_figures.xlsx` | `Fig9a_daily`, `Fig9a_summary`, `Fig9b_energy` |
| Figure 7 | `qualification_value.xlsx` | `Fig07_Daily`, `Fig07_Summary`; schedule outcomes in the other sheets |
| Figure 8 | `public_replay_figures.xlsx` | `Fig10_sensitivity` |
| Figure S1 | `controlled_figures.xlsx` | `F04_CausalTrace` |
| Figure S2 | `controlled_figures.xlsx` | `F05A_BehaviorEffect`, `F05BCD_StageCompare` |
| Figure S3 | `controlled_figures.xlsx` | `F08A_Physical`, `F08BCD_PhysFair` |
| Figure S4 | `profile_mapping.xlsx` | `Contrasts`, `Energy`, `Daily` |
| Figure S5 | `public_replay_figures.xlsx` | `Fig11_topology` |
| Figure S6 | `participant_adaptation.xlsx` | `Contrasts`, `Interaction`, `Energy`, `Daily`, `Behavior`, `Blocks` |
| Table 2 | `qualification_value.xlsx` | `Table02_Energy` (historical sheet name) |
| Table S16 | `qualification_value.xlsx` | `Seed_audit` |


Additional SI table mappings (separate from figure plotting sources):

| SI table or quantity | Workbook in `source_data/` | Worksheets and coverage |
|---|---|---|
| Table S20: bounded-record energy and contrasts | `bounded_record_feedback.xlsx` | `Energy` (totals), `Contrasts` (paired delivery differences and intervals); `Daily` retains the daily observations |
| Table S21: lower-bound minus point-record comparison | `bounded_record_feedback.xlsx` | `Contrasts` (select the named point-record comparator) |
| Table S22: replay qualification under record error | `bounded_record_feedback.xlsx` | `Recovery` |
| Table S25: PV-capacity, constrained-request and energy summary | `pv_capacity.xlsx` | `Energy`, `Contrasts`, `Exposure`; `Daily` retains the daily observations |
| Table S26: PV-capacity paired comparisons | `pv_capacity.xlsx` | `Contrasts` |
| Table S27: PV-capacity energy totals | `pv_capacity.xlsx` | `Energy` |
| Table S28: nested LV-section delivery differences and replay checks | `public_replay_figures.xlsx` | `SI_sections` (delivery differences, qualifying intervals and physical-check flags) |
| Table S29: representative LV-model delivery and replay results | `public_replay_figures.xlsx` | `Fig11_topology` (delivery differences, qualifying intervals, physical checks and voltage extrema) |

The last two rows identify the calculated-result fields supplied in those
worksheets. Structural connection/participant counts and phase or model
eligibility descriptors are given in SI Sections S13.3-S13.4 and the
repository's construction records; they are not additional columns in
`SI_sections`. The bounded-record and PV-capacity workbooks support the
separate SI tables above; they are not plotting inputs for Figure 8 or Figure S5.

All eight XLSX files are copied without recalculation. Figure 5(d) uses the
existing vector-table presentation, rounded to 0.001 kW. That presentation
was already present in the source manuscript and is not a new numerical result.

Use the current-workbook mappings above. The `INDEX` and `SOURCE_INFO` sheets inside `controlled_figures.xlsx`
retain historical figure numbers and descriptions; use this independent map
for the current article. Historical F09/F10/F11 public-study sheets
inside `controlled_figures.xlsx` are retained archive material; they are not
the source for the present Figures 6–8 or the current Australian results.
The current public results use `public_replay_figures.xlsx` and
`qualification_value.xlsx` as identified in the table.

This archive contains figure source data, not a complete simulation-input
archive. Raw third-party Ausgrid profiles and published network files remain
with their original distributors. Their identifiers and preprocessing are
described in the manuscript, supplementary information and project repository.
Editable Origin and PowerPoint masters remain in the preserved drawing workspace;
they are not necessary to compile the manuscript and are not included in Supplementary data S1.

## Current CSV companions and distribution scope

The controlled CSVs retain historical exporter names: `Fig03_*.csv` support
Figure 3; `Fig07_*.csv` support Figure 4; `Fig04_*.csv`, `Fig05_*.csv` and
`Fig08_*.csv` support Figures S1, S2 and S3. Figure 4(d)'s source
`Fig07_gate_materiality_distribution.csv` and workbook `F06D_ReplayDist`
contain 4,320 trajectory-period records, all at ECF gain 0.5. They are not
the 12,960-record replay population pooled across three gains in SI S3.5.
For Figure 8 and Figure S5, the CSV companions are respectively
`figure_data/australian_main/Fig10_sensitivity.csv` and `Fig11_topology.csv`.

The expanded qualification workbook contains Table 2's unrounded inputs
(`Table02_Energy`) and Table S16's schedule audit (`Seed_audit`). The current
twelve-schedule outputs remain in
`reproduction/outputs/ees_transition/qualification_value_12seed/`; the
original three-schedule outputs retain their historical sibling directory.

## Working in Origin

1. Open the mapped workbook or import the corresponding UTF-8 CSV. Retain control and scenario IDs as categorical columns.
2. Identify response, unit, comparator and interval type from the column header and main/SI caption before plotting.
3. Preserve paired comparisons. ECF-minus-control differences and larger-request-minus-reference differences have different baselines.
4. Use `paper/main.pdf` and `paper/supporting_information.pdf` for the current panel arrangement. Standalone artwork and manuscript compilation assets are not distributed in the current tree.
5. Plot raw observations where distributions are provided; uncertainty endpoints are not observations.

Control IDs such as `MATCHED_UNIFORM` and `DELIVERY_RATIO_CF_BETA_050`
remain machine-readable names; their manuscript names are equal-total control
and mismatch-only feedback. Preserve IDs in the data.
