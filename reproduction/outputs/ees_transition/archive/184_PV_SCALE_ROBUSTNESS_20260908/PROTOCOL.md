# PV capacity at fixed export-participant locations

## Question and fixed settings

This supplementary comparison tests whether the delivery advantage depends
on the primary installed PV scale. It does not select a new nominal case.
Four added factors are 0.50, 0.75, 1.25 and 1.50 times the primary PV scale.
The original factor 1.00 remains the primary reference. The two smaller and
two larger factors span a symmetric range around it without selecting settings
from their feedback outcomes.

Only PV size changes. The entire gross-PV chronology and PV nameplate
capacities use the same factor. Net available export is recomputed after
subtracting the unchanged household load; it is not simply multiplied.
Registered export capacities scale with PV nameplates, consistently with the
existing model. The forecast rule and its 0.90 quantile remain fixed. Its
residual increment is recomputed from the scaled 2010--2011 chronology,
using no assessment outcomes. The previous-day request and persistence
control are computed from the scaled PV and unchanged load. Thus this is a
coherent PV-capacity scenario, not an availability-record-error experiment.

Keep the original deterministic profile mapping, 418/1255 participant
locations, profile reuse, phases, topology, transformer nameplates, 80 dates,
load multiplier 0.30, power factor 0.95, gain 0.5, exact profile-derived
availability record, and daily capacity initialization. Keep the allocation
upper-voltage ceiling 1.07 pu, physical voltage range 0.90--1.10 pu,
2-W replay tolerance, scalar search and nonlinear AC implementation.
No new line ampacity is assumed. Scale neither demand nor the network ratings.

The four controls are no feedback, previous-day persistence, ECF, and each
ECF trajectory's own equal-total reduction. All four factors and all 80 dates
are retained. There are 320 new setting-days, 61,440 interval-control
records and 15,360 completed-state replays, not 320 independent dates.

## Outcomes and inference

Report each factor separately: delivered/authorized/unused energy, ECF-minus
no feedback, equal-total and persistence, daily positive/negative/unchanged
counts, qualifying intervals, source-terminal exchange, voltage and
transformer checks. Also report installed participant MW, recovery per
installed MW, and no-feedback constrained-interval counts. The latter is a
description of scarcity under the declared allocation search, not a measured
hosting-capacity threshold. Record computational effort from the original
controller logs, not from post-run audit calls.

Use 80 paired dates within each factor. Pointwise dependent wild-bootstrap
intervals use covariance max(1-|calendar-day gap|/8,0), 10,000 antithetic
replicates and key 184. Do not pool factors or replace the primary estimate
with the most favorable factor. This tests capacity at fixed adoption, not
adoption-location variation, demand-forecast error or population prevalence.

## Execution and verification

Freeze this protocol, input hashes, controls, runners and verifiers before
any new-factor day is simulated. Reproduce the first original date at factor
1.00 and retain the comparison with its previous records. Confirm that the
capture wrapper changes no controls. Independently reconstruct scaled
capacity, gross PV, available export, prior-day requests and calibrated
increments, including nonparticipant masking, before interpreting results.

Run the unchanged original scientific functions with the PV_SCALE setting
changed in the loaded runner instance only. Original source files are not
edited. The capture wrapper stores full request, award, delivery and replay
vectors and checks clipped delivery, equal-total requests and ceiling recursion.
After computation, separately evaluate each saved replay's load-only,
authorized and delivered endpoints with the same frozen nonlinear solver.
Physical failures are reported outcomes, not grounds to drop a day or
change the settings. Code/consistency errors stop the pipeline and preserve
the input and diagnostic. Resume only hash-verified completed days after
confirming an earlier owner process has ended; never duplicate a live owner.
There is no encoded maximum experiment runtime.

The primary and mapping results are already known. These four factors are
fixed before their own evaluation, not before the entire research project.
Do not insert partial results into the manuscript as complete comparisons.
