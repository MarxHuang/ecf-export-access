"""Check the prespecified longest contiguous block; do not pool incomplete jobs."""
from pathlib import Path
import json,hashlib,math,csv
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent;T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907';B=R/'169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908'
p=json.loads((A/'PROTOCOL.json').read_text('utf-8'))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
bi=max(range(len(p['blocks'])),key=lambda i:len(p['blocks'][i]));days=p['blocks'][bi]
assert len(days)==35
jobs=[j for j in p['jobs'] if j['block']==bi+1];assert len(jobs)==4
result={'scope':'Prespecified longest 35-day block only; not the complete 80-day result','dates':days,'controls':{}}
f=R/'151_FROZEN_SCENARIO_EVALUATION_20260907/runs/primary/AU_EXTERNAL_DAY_SUMMARY.csv'
assert sha(f)==p['inputs'][str(f)]
with f.open(encoding='utf-8-sig') as stream:
    nf=math.fsum(float(r['delivered_export_mwh']) for r in csv.DictReader(stream) if r['date'] in days and r['control']=='NO_FEEDBACK')
result['no_feedback_delivery_MWh']=nf
for j in jobs:
    beta=j['beta'];ecf=j['policy']=='ECF';records=[];before=None;zero_counts=[];morning_deficits=[];max_state_difference=0.;delivery=0.;reset_delivery=0.;uniform_delivery=0.
    for d in days:
        folder=A/'jobs'/j['id']/d;done=json.loads((folder/'DONE.json').read_text('utf-8'))
        assert done['protocol_sha256']==sha(A/'PROTOCOL.json')
        assert all(sha(folder/n)==h for n,h in done['files'].items())
        q=json.loads((folder/'RESULT.json').read_text('utf-8'))
        c='ECF' if ecf else 'UNQUALIFIED_SAME_UPDATE'
        records.extend([r for r in q['records'] if r['control']==c])
        with np.load(folder/'TRAJECTORIES.npz') as z:
            cap=z['capacity_full_mw'][z['participant']];start=z['start_ceiling_full_mw'][z['participant']]
            assert np.max(np.abs(start-(cap if before is None else before)))<1e-12
            before=z['end_ceiling_full_mw'][z['participant']].copy()
            delivery+=float(z['delivery'].sum())*.5
            if ecf:
                uniform_delivery+=float(z['uniform_delivery'].sum())*.5
                with np.load(T/'trajectories'/f'{d}.npz') as ref:
                    reset_delivery+=float(ref[f'beta_{beta}_delivery'].sum())*.5
                    max_state_difference=max(max_state_difference,float(np.max(np.abs(z['ceiling']-ref[f'beta_{beta}_ceiling']))))
                nz=np.flatnonzero(np.any(z['available']>0,axis=1));k=int(nz[0]) if len(nz) else 48
                for t in range(k):
                    assert np.max(np.abs(z['target'][t]-cap))<1e-12
                    assert np.max(np.abs((cap-z['ceiling'][t])-(1-beta)**t*(cap-start)))<1e-12
                after=z['ceiling'][k] if k<48 else before
                assert np.max(np.abs((cap-after)-(1-beta)**k*(cap-start)))<1e-12
                zero_counts.append(k);morning_deficits.append(float(np.max(cap-after))*1000)
            else:
                with np.load(B/'days'/d/'TRAJECTORIES.npz') as ref:reset_delivery+=float(ref['UNQUALIFIED_SAME_UPDATE_delivery'].sum())*.5
    out={'delivery_MWh':delivery,'daily_reset_delivery_MWh':reset_delivery,'carry_minus_reset_kWh':(delivery-reset_delivery)*1000,'difference_from_no_feedback_MWh':delivery-nf,'physical_failures':sum(not r['physical_pair_pass'] for r in records),'intervals':len(records)}
    if ecf:out.update(equal_total_delivery_MWh=uniform_delivery,qualifying_intervals=sum(r['gate_triggered'] for r in records),leading_zero_export_halfhours_range=[min(zero_counts),max(zero_counts)],max_first_available_ceiling_deficit_kW=max(morning_deficits),max_state_difference_from_daily_reset_kW=max_state_difference*1000)
    result['controls'][j['id']]=out
(A/'LONGEST_BLOCK_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='dates'},indent=2))
