# Figure data

Use the figure numbers in the current manuscript, not a historical export filename. The original numerical precision is retained in the CSV/XLSX files; rounding in the manuscript is for presentation.

| Current figure | Data directory and files |
|---|---|
| Fig. 1 | `figure_data/schematic/`; illustrative quantities, not experimental measurements |
| Fig. 2 | Algorithm diagram; definitions and equations in the main text, no fitted numerical series |
| Fig. 3 | `figure_data/controlled/Fig03_*.csv`, `Fig03c_request_choice_classes.xlsx` |
| Fig. 4 | `figure_data/controlled/Fig04_representative_causal_trace.csv` |
| Fig. 5 | `figure_data/controlled/Fig05_*.csv` |
| Fig. 6 | `figure_data/controlled/Fig07_*.csv` — historical data filenames use 07 for gain and replay results |
| Fig. 7 | `figure_data/controlled/Fig06_dimension_robustness_theta0.csv` — historical filename uses 06 |
| Fig. 8 | `figure_data/controlled/Fig08_*.csv` |
| Fig. 9 | `figure_data/australian_main/Fig9a_daily.csv`, `Fig9a_summary.csv`, `Fig9b_energy.csv` |
| Fig. 10 | `figure_data/australian_main/Fig10_sensitivity.csv` |
| Fig. 11 | `figure_data/australian_main/Fig11_topology.csv` |
| Fig. S1 | `figure_data/profile_mapping/` |
| Fig. S2 | `figure_data/participant_adaptation/` |

`Controlled_figure_data.xlsx` groups the controlled-study data using the original exporter organization. The separate Fig. 3(c) workbook provides the later four-category presentation. For exact category membership and counts, use the request-choice result tables and SI definitions, not inferred colours.

The `pv_capacity/` and `bounded_record/` folders support the additional SI tables. The Australian main workbook is `Fig9-11_Origin_data.xlsx`. These workbooks are included unchanged; the deposit does not regenerate or restyle them.

## Working in Origin

1. Open the workbook for the relevant family, or import the CSV with comma delimiter and UTF-8 encoding. Keep textual control and scenario IDs as categorical columns.
2. Identify the response, unit, comparator and interval type from the column header and the matching main/SI caption before selecting X, Y or error-bar columns.
3. Preserve paired comparisons. Do not mix ECF-minus-comparator differences with larger-request-minus-reference differences.
4. Use the current `paper/figures/` PDFs as the visual reference. They contain the author's final panel arrangement, not the exporter's historical figure numbering.
5. For distributions, import the underlying daily or trajectory observations where provided. Summary interval endpoints are not observations and must not be treated as a raw distribution.
6. Use Times New Roman and the author's chosen palette if matching the current artwork. Check that median, mean, central interval and uncertainty interval match the caption; they are not interchangeable.

Control IDs such as `MATCHED_UNIFORM` and `DELIVERY_RATIO_CF_BETA_050` are machine-readable names. Their manuscript names are **equal-total control** and **mismatch-only feedback**, respectively. Preserve IDs in the data and use the manuscript names in labels.
