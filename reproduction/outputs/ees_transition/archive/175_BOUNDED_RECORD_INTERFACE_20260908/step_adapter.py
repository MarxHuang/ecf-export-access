"""Connect the new bounded-record predicate to the unchanged ECF recursion."""
from pathlib import Path
import importlib.util,sys
A=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
bounded=module('bounded_record_v2',A/'v2/bounded_feedback.py')
old=module('frozen_ecf_update_151',A.parent/'151_FROZEN_SCENARIO_EVALUATION_20260907/scientific_source/src/r4r/externality_feedback.py')

def completed_feedback_step(*, capacity, ceiling_before, allocation, delivery,
                            replay_allocation, record, error_radius,
                            record_valid, replay_physical_valid, tolerance, beta):
    """One post-completion transition; no latent availability input or file read.

The caller supplies the physical predicate. This interface does not turn a
delivery lower bound into an AC uncertainty-set guarantee.
"""
    assessment=bounded.assess_bounded_record(allocation=allocation,delivery=delivery,
        replay_allocation=replay_allocation,record=record,error_radius=error_radius,
        record_valid=record_valid,replay_physical_valid=replay_physical_valid,tolerance=tolerance)
    after,target=old.update_access_ceiling(ceiling_before_mw=ceiling_before,
        registered_capacity_mw=capacity,delivery_mw=delivery,idle_indices=assessment.idle_indices,
        gate_triggered=assessment.qualifies,beta=beta)
    return assessment,after,target
