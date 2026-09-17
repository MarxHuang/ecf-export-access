"""Reconcile individual request/award vectors from completed, guarded captures.

No new allocation, power flow, controller or experimental condition is run.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

A=Path(__file__).resolve().parent
R=A.parent/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol=sha(R/'AUDIT_PROTOCOL.json')
expected_dates=json.loads((R/'AUDIT_PROTOCOL.json').read_text(encoding='utf-8'))['dates']
days=[]
intervals=[]
unrepresentable=[]
source_digests={}
for result_file in sorted((R/'days').glob('*/RESULT.json')):
    day=result_file.parent
    result=json.loads(result_file.read_text(encoding='utf-8'))
    assert result['protocol_sha256']==protocol
    for name,digest in result['outputs_sha256'].items():
        assert sha(day/name)==digest
    guard=json.loads((day/'REPRODUCTION_CLASSIFICATION.json').read_text(encoding='utf-8'))
    assert guard['scan_ready'] and guard['allocation_and_screen_outputs_match']
    assert not guard['decision_errors']
    source_digests[day.name]=sha(day/'CAPTURED_RAYS.npz')
    with np.load(day/'CAPTURED_RAYS.npz',allow_pickle=False) as f:
        req=f['request_kw']; rho=f['original_factor']; load=f['load_kw']
        assert req.shape==load.shape==(240,1255)
        assert np.all(np.isfinite(req)) and np.all(req>=0)
        assert np.all((rho>=0)&(rho<=1))
        lookup={(int(t),str(c)):k for k,(t,c) in enumerate(zip(f['interval_index'],f['control']))}
        assert set(lookup)=={(t,c) for t in range(48) for c in ('NO_FEEDBACK','PERSISTENCE_ONLY','ECF','REPLAY','MATCHED_UNIFORM')}
        for t in range(48):
            n,u,e=(lookup[t,c] for c in ('NO_FEEDBACK','MATCHED_UNIFORM','ECF'))
            assert np.array_equal(load[n],load[u]) and np.array_equal(load[n],load[e])
            rn,ru,re=req[n],req[u],req[e]
            sn,su,se=(math.fsum(v) for v in (rn,ru,re))
            w=su/sn if sn else 1.
            assert 0<=w<=1+1e-12
            proportional_error=float(np.max(np.abs(ru-w*rn)))
            assert proportional_error<1e-9
            assert abs(su-se)<1e-8
            xn,xu,xe=rho[n]*rn,rho[u]*ru,rho[e]*re
            diff=xu-xn
            signed=math.fsum(diff); l1=math.fsum(np.abs(diff))
            # Collinearity means the scalar search cannot hide large transfers
            # between participants behind a small total award difference.
            cancellation=l1-abs(signed)
            assert abs(cancellation)<1e-8
            representable=bool(rho[n]<=w+1e-14) if w>0 else bool(np.max(xn)==0)
            reduced=bool(w<1-1e-12)
            if reduced and not representable:
                unrepresentable.append({'date':day.name,'interval_index':t,
                    'uniform_factor':w,'no_feedback_factor':float(rho[n]),
                    'signed_award_difference_kw':signed})
            intervals.append({'date':day.name,'interval_index':t,
                'uniform_reduced':reduced,'original_award_representable':representable,
                'uniform_request_error_kw':proportional_error,
                'ecf_uniform_request_total_error_kw':abs(su-se),
                'uniform_no_feedback_award_signed_kw':signed,
                'uniform_no_feedback_award_l1_kw':l1,
                'uniform_no_feedback_max_connection_award_difference_kw':float(np.max(np.abs(diff))),
                'within_interval_cancellation_kw':cancellation,
                'effective_factor_difference':float(rho[u]*w-rho[n]),
                'ecf_uniform_request_l1_kw':math.fsum(np.abs(re-ru)),
                'ecf_uniform_award_signed_kw':math.fsum(xe-xu),
                'ecf_uniform_award_l1_kw':math.fsum(np.abs(xe-xu)),
            })
    days.append(day.name)

assert days
summed=lambda key:math.fsum(q[key] for q in intervals)
maximum=lambda key:max(q[key] for q in intervals)
result={'snapshot_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'completed_dates':days,'expected_days':len(expected_dates),'completed_days':len(days),
    'full_primary_coverage':set(days)==set(expected_dates),
    'half_hour_pairs':len(intervals),'protocol_sha256':protocol,
    'uniform_reduced_half_hours':sum(q['uniform_reduced'] for q in intervals),
    'reduced_and_representable':sum(q['uniform_reduced'] and q['original_award_representable'] for q in intervals),
    'unrepresentable_cases':unrepresentable,
    'max_uniform_request_vector_error_kw':maximum('uniform_request_error_kw'),
    'max_ecf_uniform_request_total_error_kw':maximum('ecf_uniform_request_total_error_kw'),
    'max_within_interval_award_cancellation_kw':maximum('within_interval_cancellation_kw'),
    'uniform_minus_no_feedback_authorization_mwh':summed('uniform_no_feedback_award_signed_kw')*.5/1000,
    'sum_absolute_connection_award_differences_mwh':summed('uniform_no_feedback_award_l1_kw')*.5/1000,
    'max_single_connection_award_difference_kw':maximum('uniform_no_feedback_max_connection_award_difference_kw'),
    'ecf_minus_uniform_authorization_mwh':summed('ecf_uniform_award_signed_kw')*.5/1000,
    'ecf_uniform_sum_absolute_connection_award_differences_mwh':summed('ecf_uniform_award_l1_kw')*.5/1000,
    'scope':'Completed captures only. Individual request and award vectors, not delivered energy. Partial sums must not replace full-study results. Equal-total is a spatial-allocation diagnostic, not an independent deployable policy or an isolated qualification test.',
    'capture_sha256':source_digests,'script_sha256':sha(Path(__file__))}
(A/'VECTOR_INTERVAL_CHECKS.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
(A/'VECTOR_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('capture_sha256','completed_dates')},ensure_ascii=False,indent=2))
