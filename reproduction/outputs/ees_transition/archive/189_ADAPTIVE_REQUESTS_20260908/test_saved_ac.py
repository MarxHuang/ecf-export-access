"""Boundary tests for second-execution reporting; no experimental file edits."""
from pathlib import Path
import copy,math
from check_saved_ac import screen_pass
from run_adaptation import A,verify,sha,write

verify()
good=dict(converged=True,customer_undervoltage_count=0,customer_overvoltage_count=0,
          transformer_overload_count=0,physical_pair_pass=True,
          customer_voltage_min_pu=.9,customer_voltage_max_pu=1.07,transformer_max_loading_pu=1.)
accepted=[];rejected=[]
def accept(label,expected,authorization=False,**kw):
    s=dict(good,**kw);old=copy.deepcopy(s)
    assert screen_pass(s,authorization) is expected,label
    assert s==old;accepted.append(label)
def reject(label,**kw):
    try:screen_pass(dict(good,**kw))
    except AssertionError:rejected.append(label)
    else:raise AssertionError(label)
accept('authorized_upper_boundary',True,True)
accept('delivery_boundary',True,customer_voltage_max_pu=1.1)
accept('authorization_stricter_than_delivery',False,True,customer_voltage_max_pu=1.08)
accept('transformer_tolerance',True,transformer_max_loading_pu=1+5e-10)
accept('undervoltage_retained',False,customer_voltage_min_pu=.89,customer_undervoltage_count=1,physical_pair_pass=False)
accept('overvoltage_retained',False,customer_voltage_max_pu=1.11,customer_overvoltage_count=1,physical_pair_pass=False)
accept('overload_retained',False,transformer_max_loading_pu=1.01,transformer_overload_count=1,physical_pair_pass=False)
accept('nonconvergence_retained',False,converged=False,physical_pair_pass=False)
reject('hidden_undervoltage',customer_voltage_min_pu=.89)
reject('hidden_overvoltage',customer_voltage_max_pu=1.11)
reject('hidden_overload',transformer_max_loading_pu=1.01)
reject('false_failure_flag',physical_pair_pass=False)
reject('false_success_flag',converged=False)
reject('reversed_extrema',customer_voltage_min_pu=1.09)
reject('nan_voltage',customer_voltage_max_pu=math.nan)
reject('nan_loading',transformer_max_loading_pu=math.nan)
reject('false_violation_count',customer_overvoltage_count=1,physical_pair_pass=False)
verify()
r=dict(passed=True,accepted_screens=accepted,rejected_contradictions=rejected,
       checker_sha256=sha(A/'check_saved_ac.py'),test_sha256=sha(Path(__file__)))
write(A/'SAVED_AC_TESTS.json',r);print(r)
