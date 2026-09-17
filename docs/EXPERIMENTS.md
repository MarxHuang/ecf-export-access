# Experiment and version map

All paths below are relative to `reproduction/`. The main text and SI in `paper/` are the authority for scientific definitions and figure numbers.

## Controlled study

- `outputs/endogenous_request_choice_v14/`: request-choice surfaces and finite-difference diagnostics used in Fig. 3.
- `outputs/externality_conditioned_feedback_v16/`: interval and trajectory results, gain and control comparisons, classification diagnostics, and allocation/delivery AC checks used in Figs. 4–8 and the corresponding SI tables.
- `outputs/rate_v4_design/V4_FREEZE_DEVELOPMENT/`: the frozen synthetic branch-rating policy and calibrated allocation inputs.
- `tools/run_v16_externality_conditioned_feedback.py`: experiment runner. It imports some helpers with older version identifiers; those imports are preserved and included.

`V16_DESIGN.json` retains source metadata inherited from earlier scenario construction. A legacy `report_multiplier` field must not be interpreted as the endogenous multiplier selected in every reported experiment. Request-choice outputs and the runner's actual behavior inputs determine those requests.

## Australian studies

These paths are under `outputs/ees_transition/archive/`:

| Directory prefix | Role in the current paper |
|---|---|
| `151_FROZEN_SCENARIO_EVALUATION_20260907` | Primary 80-day comparison, gain/record/cohort sensitivities, nested sections, and representative LV models; Figs. 9–11 |
| `157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2` | One-dimensional allocation-search checks |
| `160_ALLOCATION_VECTOR_REVIEW_20260907` | Allocation-vector diagnostics |
| `164_CEILING_TRAJECTORY_REVIEW_20260907` | Ceiling-trajectory checks |
| `169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908` | Same update formula with and without the qualifying comparison |
| `173_CONTIGUOUS_STATE_FEEDBACK_20260908` | State retention across contiguous observed days and recovery diagnostics |
| `175_BOUNDED_RECORD_INTERFACE_20260908` | Availability-record uncertainty interface |
| `176_BOUNDED_RECORD_FEEDBACK_20260908` | Feedback under bounded record error |
| `180_PROFILE_MAPPING_ROBUSTNESS_20260908` | Alternative profile placements; Fig. S1 |
| `184_PV_SCALE_ROBUSTNESS_20260908` | PV-capacity sensitivity |
| `185_INDIVIDUAL_DELIVERY_REVIEW_20260908` | Distribution of individual delivery changes |
| `186_ADAPTATION_INFORMATION_REVIEW_20260908` | Information available to the participant-response model |
| `189_ADAPTIVE_REQUESTS_20260908` | Experience-based request adjustment; Fig. S2 |

The Australian scientific modules are preserved in `151_.../scientific_source/`. They must not be replaced by similarly named modules in the controlled-study source tree. The latter tree also contains preprocessing utilities; the frozen implementation is the authority for the current Australian numerical results.

The deposit includes the existing experiment code, protocols and aggregate/interval tables where listed in the manifest. Large auxiliary caches containing all captured solver rays and per-connection trajectory vectors are not included. Several legacy diagnostic runners expect these caches and original absolute paths. Their source is supplied for inspection; they are not advertised as portable, one-command reproductions. The new primary-case launcher only relocates input/output paths and leaves numerical settings and the frozen source unchanged.

The current Australian suite uses load multiplier 0.30 and an authorization voltage ceiling of 1.07 pu. Its physical assessment bounds remain those stated in the manuscript. `EVALUATION_PROTOCOL.json` records the chronology of scenario design, including prior evaluation exposure; it should not be described as a newly blinded experiment.
