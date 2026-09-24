"""Prepare the verified Origin-ready data for NC Fig. 7 and Table 2."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


HERE = Path(__file__).resolve().parent
NC_ROOT = HERE.parents[1]
OUTPUTS = HERE / "results_12seed"
SUMMARY = json.loads((OUTPUTS / "SUMMARY.json").read_text(encoding="utf-8"))
CALENDAR = json.loads((HERE / "results" / "CALENDAR_ROBUSTNESS.json").read_text(encoding="utf-8"))
ARCHIVE = NC_ROOT.parents[1] / "ees_transition" / "archive"
SAME_UPDATE = json.loads((ARCHIVE / "169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908"
                          / "VERIFIED_COMPARISON.json").read_text(encoding="utf-8"))
PRIMARY = json.loads((ARCHIVE / "151_FROZEN_SCENARIO_EVALUATION_20260907"
                      / "runs" / "primary" / "AU_EXTERNAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
SEEDS = tuple(SUMMARY["seeds"])


with (OUTPUTS / "DAILY_RESULTS.csv").open(encoding="utf-8-sig", newline="") as stream:
    daily = list(csv.DictReader(stream))
dates = sorted(set(row["date"] for row in daily))
assert len(dates) == 80 and len(daily) == 80 * len(SEEDS)
by_day = {d: {int(row["seed"]): row for row in daily if row["date"] == d} for d in dates}
daily_values = []
for j, d in enumerate(dates, 1):
    rows = by_day[d]
    ecf = float(rows[SEEDS[0]]["ecf_delivered_export_mwh"])
    no_feedback = float(rows[SEEDS[0]]["no_feedback_delivered_export_mwh"])
    random = [float(rows[seed]["delivered_export_mwh"]) for seed in SEEDS]
    assert all(abs(float(rows[seed]["ecf_delivered_export_mwh"]) - ecf) < 1e-10 for seed in SEEDS)
    daily_values.append({
        "date": d, "day_index": j, "ecf_mwh": ecf, "no_feedback_mwh": no_feedback,
        **{f"random_{seed}_mwh": value for seed, value in zip(SEEDS, random)},
        "random_mean_mwh": float(np.mean(random)),
        "ecf_minus_random_mean_kwh": float((ecf - np.mean(random)) * 1000),
        "ecf_minus_no_feedback_kwh": (ecf - no_feedback) * 1000,
        "longest_35_day_block": int("2013-05-16" <= d <= "2013-06-19"),
    })
assert abs(sum(row["ecf_minus_random_mean_kwh"] for row in daily_values) / 1000 -
           SUMMARY["ecf_minus_random_seed_average_mwh"]) < 1e-8

no_feedback_total = SUMMARY["no_feedback_delivered_mwh"]
random_totals = [SUMMARY["seed_results"][str(seed)]["random_delivered_mwh"] for seed in SEEDS]
random_updates = [SUMMARY["seed_results"][str(seed)]["random_update_count"] for seed in SEEDS]
ecf_total = SUMMARY["ecf_delivered_mwh"]
unqualified_total = SAME_UPDATE["totals"]["UNQUALIFIED_SAME_UPDATE"]["delivery"]
unqualified_updates = SAME_UPDATE["counts"]["UNQUALIFIED_SAME_UPDATE_update_intervals"]
assert abs(SAME_UPDATE["totals"]["ECF"]["delivery"] - ecf_total) < 1e-9
assert SAME_UPDATE["counts"]["ECF_update_intervals"] == 483

book = Workbook()
sheet = book.active
sheet.title = "Fig07_Daily"
daily_headers = list(daily_values[0])
sheet.append(daily_headers)
for row in daily_values:
    sheet.append([row[name] for name in daily_headers])
sheet.freeze_panes = "C2"
sheet.auto_filter.ref = sheet.dimensions

sheet = book.create_sheet("Fig07_Summary")
sheet.append(["control", "update_intervals", "delivered_80d_MWh", "minus_no_feedback_MWh", "note"])
sheet.append(["No feedback", 0, no_feedback_total, 0, "Fixed connection ceiling"])
for seed, delivered, updates in zip(SEEDS, random_totals, random_updates):
    sheet.append([f"Randomized {seed}", updates, delivered, delivered - no_feedback_total,
                  "Hash-based schedule; same target and gain as ECF"])
sheet.append(["Randomized mean", np.mean(random_updates), np.mean(random_totals),
              np.mean(random_totals) - no_feedback_total, "Twelve fixed seeds averaged within date"])
sheet.append(["ECF", 483, ecf_total, ecf_total - no_feedback_total,
              "Replay-qualified update"])
sheet.append(["Update every unused award", unqualified_updates, unqualified_total,
              unqualified_total - no_feedback_total, "Same target and gain; no replay test"])

sheet = book.create_sheet("Table02_Energy")
sheet.append(["control", "authorized_80d_MWh", "unused_80d_MWh",
              "delivered_80d_MWh", "updated_half_hours", "source"])
for label, original_key in (("No feedback", "NO_FEEDBACK"),
                            ("Equal-total reduction", "MATCHED_UNIFORM")):
    r = PRIMARY["aggregate"][original_key]
    sheet.append([label, r["authorized_export_mwh"], r["idle_authorization_mwh"],
                  r["delivered_export_mwh"], 0, "Frozen primary simulation"])
r = SAME_UPDATE["totals"]["UNQUALIFIED_SAME_UPDATE"]
sheet.append(["Update without replay test", r["authorization"], r["unused"],
              r["delivery"], unqualified_updates, "Unqualified-update comparison"])
sheet.append(["Frequency-calibrated random update", np.mean([
    SUMMARY["seed_results"][str(seed)]["authorized_export_mwh"] for seed in SEEDS]),
    np.mean([SUMMARY["seed_results"][str(seed)]["idle_authorization_mwh"] for seed in SEEDS]),
    np.mean(random_totals), np.mean(random_updates), "Twelve fixed schedules averaged"])
r = SAME_UPDATE["totals"]["ECF"]
sheet.append(["ECF", r["authorization"], r["unused"], r["delivery"], 483,
              "Frozen primary simulation"])

sheet = book.create_sheet("Seed_audit")
headers = ["seed", "update_intervals", "eligible_idle_intervals", "authorized_MWh",
           "unused_MWh", "delivered_MWh", "ECF_minus_seed_MWh",
           "positive_days", "negative_days", "allocation_pass", "physical_pass", "AC_solve_count"]
sheet.append(headers)
for seed in SEEDS:
    r = SUMMARY["seed_results"][str(seed)]
    sheet.append([seed, r["random_update_count"], r["eligible_unused_intervals"],
                  r["authorized_export_mwh"], r["idle_authorization_mwh"],
                  r["random_delivered_mwh"], r["ecf_minus_random_mwh"],
                  r["positive_days"], r["negative_days"], int(r["all_allocation_pass"]),
                  int(r["all_physical_pass"]), r["power_flow_count"]])

sheet = book.create_sheet("Calendar_audit")
sheet.append(["comparison", "correlation_support_days", "daily_mean_kWh",
              "95pct_lower_kWh", "95pct_upper_kWh", "longest35_MWh", "other45_MWh"])
for name, entry in CALENDAR["comparisons"].items():
    for support, ci in entry["bandwidth_days_and_95pct_mean_kwh"].items():
        sheet.append([name, int(support), entry["daily_mean_kwh"], ci[0], ci[1],
                      entry["longest_35_day_block_mwh"], entry["remaining_45_day_total_mwh"]])

sheet = book.create_sheet("Read_me")
for line in [
    "The new NC Fig. 7 has one sheet with daily panel data and one with total panel data.",
    "Table02_Energy contains exact unrounded inputs to main Table 2.",
    "The randomized control was specified after the original results and is a diagnostic, not an independent validation.",
    "All twelve fixed schedules are retained; do not select a seed by outcome.",
    "Fig. 7(a) uses ecf_minus_random_mean_kwh; dashed vertical lines bound the longest 35-day block.",
    "Fig. 7(b) uses minus_no_feedback_MWh and update_intervals from Fig07_Summary.",
    "The original published control and AC results were not recomputed for this figure.",
]:
    sheet.append([line])

for worksheet in book:
    worksheet.sheet_view.showGridLines = False
    for cell in worksheet[1]:
        cell.font = Font(name="Times New Roman", bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="295C70")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    worksheet.row_dimensions[1].height = 30
    for col in worksheet.columns:
        letter = col[0].column_letter
        worksheet.column_dimensions[letter].width = min(37, max(14, max(len(str(c.value or "")) for c in col) + 3))
        for cell in col[1:]:
            cell.font = Font(name="Times New Roman", size=10)
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"

workbook_path = NC_ROOT / "source_data" / "qualification_value.xlsx"
book.save(workbook_path)
print(json.dumps({"workbook": str(workbook_path),
                  "source_rows_daily": len(daily_values), "random_update_count": random_updates},
                 ensure_ascii=False))
