"""Vectorized second implementation, without importing the decision interface."""
from pathlib import Path
import hashlib,json
from collections import Counter
import numpy as np
A=Path(__file__).resolve().parent;V=A/'v2';R=A.parent
T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907/trajectories'
H=R/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2/days'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((V/'PROTOCOL.json').read_text('utf-8'))
r=json.loads((V/'FIXED_PAIR_RESULTS.json').read_text('utf-8'))
assert r['protocol_sha256']==sha(V/'PROTOCOL.json')
for n,h in p['inputs'].items():assert sha(Path(n))==h,n
for n,h in p['implementation'].items():assert sha(V/n)==h,n
indexed={(q['date'],q['interval'],q['radius_capacity_fraction'],q['bias_sign']):q for q in r['rows']}
assert len(indexed)==23040
cnt=Counter();maxerr=0.;matched=0
for day in sorted({q['date'] for q in r['rows']}):
    with np.load(T/f'{day}.npz') as z:
        x=z['beta_0.5_award'];y=np.minimum(z['beta_0.5_delivery'],x)
        a=z['available_mw'];C=z['capacity_mw']
    with np.load(H/day/'CAPTURED_RAYS.npz') as z:
        ids={int(t):k for k,t in enumerate(z['interval_index']) if z['control'][k]=='REPLAY'}
        xr=np.stack([z['request_kw'][ids[t],z['participant']]*z['original_factor'][ids[t]]/1000 for t in range(48)])
    idle=x-y>p['tolerance_MW'];outside=~idle
    access=np.maximum(xr-x,0)*outside
    def evaluate(yr):
        outside_gain=np.sum(np.maximum(yr-y,0)*outside,axis=1)
        total=np.sum(yr-y,axis=1)
        passes=idle.any(axis=1)&(access.sum(axis=1)>p['tolerance_MW'])&(outside_gain>p['tolerance_MW'])&(total>p['tolerance_MW'])
        return passes,outside_gain,total
    truth=evaluate(np.minimum(xr,a))[0]
    for fraction in p['fractions']:
        delta=fraction*C
        for sign in p['signs']:
            record=np.maximum(y,a+sign*delta)
            point=evaluate(np.minimum(xr,record))
            lower=evaluate(np.minimum(xr,np.maximum(y,record-delta)))
            assert not np.any(lower[0]&~truth)
            for t in range(48):
                q=indexed[day,t,fraction,sign];assert bool(truth[t])==q['true_qualification']
                for key,values in [('POINT',point),('LOWER_BOUND',lower)]:
                    assert bool(values[0][t])==q[key]['qualifies']
                    for computed,field in [(values[1][t],'outside_delivery_MW'),(values[2][t],'total_delivery_MW')]:
                        error=abs(computed-q[key][field]);maxerr=max(maxerr,error);assert error<1e-12
                    cls=('true_positive' if truth[t] else 'false_positive') if values[0][t] else ('false_negative' if truth[t] else 'true_negative')
                    cnt[fraction,sign,key,cls]+=1
                matched+=1
for q in r['summary']:
    for k in ['true_positive','false_positive','false_negative','true_negative']:
        assert q[k]==cnt[q['radius_capacity_fraction'],q['bias_sign'],q['method'],k]
assert matched==23040
out={'all_perturbed_pairs_independently_checked':matched,'method_comparisons_checked':2*matched,
     'max_gain_arithmetic_difference_MW':maxerr,'no_false_positive_lower_bound':True,
     'source_inputs_unchanged':True,'results_sha256':sha(V/'FIXED_PAIR_RESULTS.json'),
     'scope':'Same fixed pairs and declared error levels; does not add field data or a temporal control run.'}
(A/'INDEPENDENT_CHECK.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
