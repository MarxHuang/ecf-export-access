"""Record-only replay and bounded recovery, with no latent availability input."""
from pathlib import Path
import importlib.util,sys
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
core=load_module('record_bound_core_175',R/'175_BOUNDED_RECORD_INTERFACE_20260908/v2/bounded_feedback.py')

def screen_record_delivery(*,model,load_kw,net_import_kw,replay_delivery_mw,participant):
    """PCC power is metered residual import minus candidate replay export.

Only record-derived candidate export enters this call. Household Q is held
fixed by the original model. Gross PV is a numerical encoding of that PCC
state, not a latent or measured-availability input to the decision.
"""
    export=np.asarray(replay_delivery_mw)*1000
    imports=np.asarray(net_import_kw)
    if np.any((export>1e-9)&(imports>1e-9)&participant):
        raise ValueError('Record-derived export conflicts with metered positive import')
    encoded_pv=np.where(participant,np.maximum(np.asarray(load_kw)-imports+export,0.),0.)
    screen,delivered,_=model.solve(load_kw,encoded_pv,export,participant)
    assert np.max(np.abs(delivered-export))<1e-9
    return screen

def choose_transition(*,runner,model,load_kw,net_import_kw,participant,request,
                      allocation,delivery,capacity,ceiling,record,error_radius,mode):
    """One completed-interval decision; arguments exclude truth and raw PV."""
    tol=runner.REPLAY_TOLERANCE_MW
    replay_request,idle=runner.fulfillment_replay_request(request,allocation,delivery,tolerance_mw=tol)
    replay=runner.allocate_proportional_ac_prefix(model,load_kw,np.asarray(replay_request)*1000,
        participant,scalar_tolerance=runner.SCALAR_TOLERANCE,
        authorization_voltage_max_pu=runner.AUTHORIZATION_VOLTAGE_MAX_PU)
    xr=replay.allocation_kw/1000
    predicted=np.minimum(xr,record)
    ps=screen_record_delivery(model=model,load_kw=load_kw,net_import_kw=net_import_kw,
        replay_delivery_mw=predicted,participant=participant)
    physical=bool(replay.pair_pass and ps.physical_pair_pass)
    radius=np.asarray(error_radius) if mode=='LOWER_BOUND' else np.zeros_like(error_radius)
    assert mode in ('LOWER_BOUND','POINT')
    q=core.assess_bounded_record(allocation=allocation,delivery=delivery,replay_allocation=xr,
        record=record,error_radius=radius,record_valid=True,replay_physical_valid=physical,tolerance=tol)
    if not q.record_valid:raise ValueError(q.reason)
    after,target=runner.update_access_ceiling(ceiling_before_mw=ceiling,
        registered_capacity_mw=capacity,delivery_mw=delivery,idle_indices=idle,
        gate_triggered=q.qualifies,beta=.5)
    return dict(assessment=q,after=np.asarray(after),target=np.asarray(target),
        replay_allocation=xr,replay_request=np.asarray(replay_request),
        record_replay_delivery=predicted,replay_record_screen=ps,
        replay_allocation_result=replay,replay_physical_pass=physical,
        extra_power_flows=replay.power_flow_count+1)
