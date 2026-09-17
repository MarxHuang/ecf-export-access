"""Check the disclosed controller inputs and record-only physical encoding."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,inspect,json
import numpy as np
from control_step import choose_transition,screen_record_delivery,load_module
A=Path(__file__).resolve().parent;R=A.parent
old=load_module('unit_old_ecf',R/'151_FROZEN_SCENARIO_EVALUATION_20260907/scientific_source/src/r4r/externality_feedback.py')
class Model:
    def solve(self,load,pv,export,mask):
        self.last_pv=np.asarray(pv).copy()
        available=np.maximum(np.where(mask,pv,0.)-load,0.)
        delivered=np.minimum(export,available)
        return SimpleNamespace(physical_pair_pass=True),delivered,available
def allocator(model,load,request,mask,**kwargs):
    v=np.asarray(request);v=v*min(1.,3./v.sum())
    return SimpleNamespace(allocation_kw=v,pair_pass=True,power_flow_count=1)
m=SimpleNamespace(REPLAY_TOLERANCE_MW=2e-6,SCALAR_TOLERANCE=1e-6,AUTHORIZATION_VOLTAGE_MAX_PU=1.07,
    fulfillment_replay_request=old.fulfillment_replay_request,update_access_ceiling=old.update_access_ceiling,
    allocate_proportional_ac_prefix=allocator)
args=dict(runner=m,model=Model(),load_kw=np.array([1.,1.]),net_import_kw=np.zeros(2),
    participant=np.array([True,True]),request=np.array([.002,.002]),allocation=np.array([.002,.001]),
    delivery=np.array([.001,.001]),capacity=np.array([.003,.003]),ceiling=np.array([.003,.003]),
    record=np.array([.001,.002]),error_radius=np.array([.0005,.0005]))
point=choose_transition(**args,mode='POINT');bound=choose_transition(**args,mode='LOWER_BOUND')
assert point['assessment'].qualifies and bound['assessment'].qualifies
assert bound['assessment'].total_delivery_gain<point['assessment'].total_delivery_gain
assert np.array_equal(point['record_replay_delivery'],bound['record_replay_delivery'])
assert np.allclose(bound['after'],[.002,.003],rtol=0,atol=1e-15)
assert np.max(np.abs(args['model'].last_pv-(args['load_kw']+bound['record_replay_delivery']*1000)))<1e-12
try:choose_transition(**args,mode='POINT',true_available=np.array([99.,99.]))
except TypeError:pass
else:raise AssertionError('Latent availability accepted')
try:screen_record_delivery(model=Model(),load_kw=np.ones(2),net_import_kw=np.ones(2),replay_delivery_mw=np.array([.001,0.]),participant=np.ones(2,dtype=bool))
except ValueError:pass
else:raise AssertionError('Contradictory import/export accepted')
sig=inspect.signature(choose_transition)
assert not any(s in sig.parameters for s in ['actual_a','gross_pv_kw','true_available','actual_screen'])
tree=ast.parse(inspect.getsource(choose_transition))
names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
assert not names&{'actual_a','gross_by_day','_WORKER','npz','truth'}
# Changing an irrelevant latent field on the mock has no effect on the result.
m.latent_available=np.array([1e9,1e9])
again=choose_transition(**args,mode='LOWER_BOUND')
assert np.array_equal(again['after'],bound['after'])
out={'record_only_signature':True,'unexpected_truth_argument_rejected':True,
     'point_and_lower_share_record_based_physical_state':True,
     'contradictory_import_export_rejected':True,'synthetic_step_formula_check':True,
     'controller_has_no_observed_truth_variable_access':True,
     'control_step_sha256':hashlib.sha256((A/'control_step.py').read_bytes()).hexdigest()}
(A/'INFORMATION_PATH_TESTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
