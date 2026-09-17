"""Capacity-target recovery and request-binding checks after full verification."""
from pathlib import Path
import json,hashlib
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent;T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907'
p=json.loads((A/'PROTOCOL.json').read_text('utf-8'))
v=json.loads((A/'VERIFIED_RESULTS.json').read_text('utf-8'));assert v['complete']
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert v['protocol_sha256']==sha(A/'PROTOCOL.json')
rows=[];binding=[];capacity_recovery_checks=0;max_recovery_error=0.
for job in p['jobs']:
    if job['policy']!='ECF':continue
    beta=job['beta']
    runs=np.zeros(418,dtype=int);longest=np.zeros(418,dtype=int)
    recovery_steps=0;recovery_start=None
    for day_index,d in enumerate(job['dates']):
        folder=A/'jobs'/job['id']/d
        assert all(sha(folder/n)==h for n,h in v['output_hashes'][job['id']+'/'+d].items())
        with np.load(folder/'TRAJECTORIES.npz') as z, np.load(T/'trajectories'/f'{d}.npz') as ref:
            a=z['available'];ceil=z['ceiling'];cap=z['capacity_full_mw'][z['participant']];target=z['target']
            assert np.max(z['raw_request_mw']-cap)<1e-10
            for t in range(48):
                binds=z['raw_request_mw'][t]-z['request'][t]>2e-6
                runs=np.where(binds,runs+1,0);longest=np.maximum(longest,runs)
                if np.array_equal(target[t],cap):
                    if recovery_steps==0:recovery_start=ceil[t].copy()
                    recovery_steps+=1
                    after=ceil[t+1] if t<47 else z['end_ceiling_full_mw'][z['participant']]
                    e=float(np.max(np.abs((cap-after)-(1-beta)**recovery_steps*(cap-recovery_start))))
                    assert e<1e-12
                    max_recovery_error=max(max_recovery_error,e);capacity_recovery_checks+=1
                else:recovery_steps=0;recovery_start=None
            nonzero=np.flatnonzero(np.any(a>0,axis=1));k=int(nonzero[0]) if len(nonzero) else 48
            delta0=ceil[0]-ref[f'beta_{beta}_ceiling'][0]
            for t in range(k):
                assert np.max(np.abs(target[t]-cap))<1e-12
                assert np.max(np.abs((cap-ceil[t])-(1-beta)**t*(cap-ceil[0])))<1e-12
                assert np.max(np.abs((ceil[t]-ref[f'beta_{beta}_ceiling'][t])-(1-beta)**t*delta0))<1e-12
            remaining=ceil[k] if k<48 else z['end_ceiling_full_mw'][z['participant']]
            assert np.max(np.abs((cap-remaining)-(1-beta)**k*(cap-ceil[0])))<1e-12
            row={'block':job['block'],'date':d,'beta':beta,'day_in_block':day_index+1,'leading_zero_export_halfhours':k,'midnight_max_deficit_kW':float(np.max(cap-ceil[0]))*1000,'after_zero_export_max_deficit_kW':float(np.max(cap-remaining))*1000}
            if k<48:row['first_available_max_state_difference_from_daily_reset_kW']=float(np.max(np.abs(ceil[k]-ref[f'beta_{beta}_ceiling'][k])))*1000
            rows.append(row)
    binding.append({'block':job['block'],'beta':beta,'max_binding_halfhours':int(longest.max()),'participant_longest_binding_halfhours_quantiles':np.quantile(longest,[0,.5,.95,1]).tolist(),'definition':'Raw request minus admitted request greater than the existing 2 W tolerance; episodes stop on unbound intervals or calendar gaps.'})
out={'protocol_sha256':sha(A/'PROTOCOL.json'),'checked_days':len(rows),'rows':rows,'request_binding':binding,'capacity_recovery_checks':capacity_recovery_checks,'max_capacity_recovery_identity_error_MW':max_recovery_error,'scope':'Closed-condition recovery only; no global feedback contraction or delivery bound is inferred.'}
(A/'RECOVERY_DIAGNOSTIC.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified zero-export recovery on',len(rows),'gain-day pairs')
