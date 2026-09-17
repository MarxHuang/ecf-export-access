# Profile placement with unchanged uptake and chronology

This supplementary experiment tests whether the primary comparison depends
on the particular household-profile-to-service-point association. It does not
replace the primary scenario or search for a favourable assignment.

Five permutations, labelled M01–M05, use the fixed keys 18001–18005. For each
participant-status group separately, sort the service records by load name.
Sort the donor records by SHA256 of `key|group|load_name`, breaking ties by load
name, and assign their customer-profile identifiers to the ordered recipients.
Only `ausgrid_customer_id` changes. The participant locations, 418/1255 uptake,
network topology, phase connections, and all adoption metadata remain fixed.
The entire profile, its capacity, and its forecast-calibration history move
together. Each group's profile multiset is unchanged, including repeated
profiles. Aggregate exogenous chronology and installed participant capacity are
therefore unchanged. This is not a test of adoption-location uncertainty,
penetration severity, independently sampled feeders, or field deployment.

Use all original 80 eligible dates, exact modeled available-export records,
gain 0.5, daily capacity initialization, load scale 0.30, unchanged PV scale,
allocation voltage ceiling 1.07 pu, stated physical voltage limits 0.90–1.10 pu
and published transformer ratings. Reuse the frozen round151 runner and core
without edits. Four original controls are no feedback, previous-day
persistence, ECF, and each ECF trajectory's equal-total control. There is no new
same-update-without-qualification comparator in this experiment.

Report every mapping separately: delivered/authorized/unused energy, ECF minus
no feedback and equal-total, daily sign counts, qualification counts, source
exchange, voltage and transformer violations. Record persistence as a secondary
comparison. Use 80 paired dates within a mapping for date-dependent intervals,
not 400 independent days or 5 independent feeders. If intervals are computed,
reuse the calendar kernel max(1−|day gap|/8,0), 10,000 antithetic wild replicates
and seed 180. Pointwise intervals do not carry simultaneous coverage claims.

Freeze source and mapping hashes before any new-mapping power flow. Do not
change maps, parameters, dates, or limits after seeing results. Retain physical
failures and adverse effects; distinguish delivery differences from trajectories
passing the declared physical checks. No experiment runtime deadline is encoded.
An interrupted run may resume only completed days with verified hashes; a
still-live job must not be duplicated. Code/consistency errors stop processing
and preserve the offending input; a reported physical failure is an outcome,
not grounds for changing the scenario.

Before the full run, reproduce the first original date and check saved results
against the published primary records. Instrumentation captures requests,
awards, deliveries, reference availability, replay checks and ceiling updates,
but never changes the original decisions. Independently verify the clipped
delivery, update recursion, request equality and recovery inequalities for
each completed day. A completed-date scientific exception is not silently
discarded or replaced.

The primary results are already known. These maps are fixed before their own
evaluation, not before the entire research project. No claim of a new globally
blind validation is made.
