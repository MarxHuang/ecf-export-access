# Figure data

Use the figure numbers in the current manuscript, not a historical export filename. The original numerical precision is retained in the CSV/XLSX files; rounding in the manuscript is for presentation.

| Current figure | Data directory and files |
|---|---|
| Fig. 1 | `figure_data/schematic/`; illustrative quantities, not experimental measurements |
| Fig. 2 | Algorithm diagram; definitions and equations in the main text, no fitted numerical series |
| Fig. 3 | `figure_data/controlled/Fig03_*.csv`, `Fig03c_request_choice_classes.xlsx` |
| Fig. 4 | `figure_data/controlled/Fig07_*.csv` — historical data filenames use 07 for gain and replay results |
| Fig. 5 | `figure_data/nc_submission/controlled_figures.xlsx`, sheet `F07_Robustness`; `figure_data/controlled/Fig06_dimension_robustness_theta0.csv` holds a related analysis |
| Fig. 6 | `figure_data/australian_main/Fig9a_daily.csv`, `Fig9a_summary.csv`, `Fig9b_energy.csv` |
| Fig. 7 | `figure_data/qualification_value/qualification_value.xlsx`: `Fig07_Daily`, `Fig07_Summary` |
| Fig. 8 | `figure_data/australian_main/Fig10_sensitivity.csv` |
| Supplementary Fig. 1 | `figure_data/controlled/Fig04_representative_causal_trace.csv` |
| Supplementary Fig. 2 | `figure_data/controlled/Fig05_*.csv` |
| Supplementary Fig. 3 | `figure_data/controlled/Fig08_*.csv` |
| Supplementary Fig. 4 | `figure_data/profile_mapping/` |
| Supplementary Fig. 5 | `figure_data/australian_main/Fig11_topology.csv` |
| Supplementary Fig. 6 | `figure_data/participant_adaptation/` |

`figure_data/controlled/Controlled_figure_data.xlsx` retains the original exporter organization. The current submission's consolidated workbook is `figure_data/nc_submission/controlled_figures.xlsx`; its tab names differ from the original exporter. Use `F02A_UtilityCurve`, `F02BC_ChoiceSurface` and `F02D_ConsequenceEq` for Fig. 3, `F06AB_Beta`, `F06C_PairedTargets` and `F06D_ReplayDist` for Fig. 4, and `F07_Robustness` for Fig. 5. The separate Fig. 3(c) workbook provides the four-category presentation. For exact category membership and counts, use the request-choice result tables and SI definitions, not inferred colours.

For Supplementary Figs. 1-3, the consolidated tabs are `F04_CausalTrace`, `F05A_BehaviorEffect`/`F05BCD_StageCompare`, and `F08A_Physical`/`F08BCD_PhysFair`, respectively. Historical F09-F11 tabs in this consolidated workbook are not the source for the current Australian figures. Use the dedicated `australian_main/` and `qualification_value/` files listed above for main Figs. 6-8 and Supplementary Fig. 5.

The `pv_capacity/` and `bounded_record/` folders support additional SI tables.
The Australian historical workbook is `Fig9-11_Origin_data.xlsx`; its sheet
names keep the older figure numbers. The expanded diagnostic workbook also
contains exact unrounded inputs to main Table 2 (`Table02_Energy`) and
Supplementary Table 16 (`Seed_audit`). The current 12-schedule daily and
interval outputs are in
`reproduction/outputs/ees_transition/qualification_value_12seed/`; the
original three-schedule record is preserved in the sibling directory without
the suffix.

## Working in Origin

1. Open the workbook for the relevant family, or import the CSV with comma delimiter and UTF-8 encoding. Keep textual control and scenario IDs as categorical columns.
2. Identify the response, unit, comparator and interval type from the column header and the matching main/SI caption before selecting X, Y or error-bar columns.
3. Preserve paired comparisons. Do not mix ECF-minus-comparator differences with larger-request-minus-reference differences.
4. Use the figures embedded in `paper/main.pdf` and `paper/supporting_information.pdf` as the visual reference. Standalone manuscript figure files and TeX compilation assets are not distributed in the current tree. Follow the current panel arrangement, not the exporter's historical figure numbering.
5. For distributions, import the underlying daily or trajectory observations where provided. Summary interval endpoints are not observations and must not be treated as a raw distribution.
6. Use Times New Roman and the author's chosen palette if matching the current artwork. Check that median, mean, central interval and uncertainty interval match the caption; they are not interchangeable.

Control IDs such as `MATCHED_UNIFORM` and `DELIVERY_RATIO_CF_BETA_050` are machine-readable names. Their manuscript names are **equal-total control** and **mismatch-only feedback**, respectively. Preserve IDs in the data and use the manuscript names in labels.
