"""Bounded-record tests of every original replay pair; not a new feedback run."""
from pathlib import Path
import argparse,csv,hashlib,json,time
from collections import Counter
import numpy as np
from bounded_feedback import assess_bounded_record as assess
A=Path(__file__).resolve().parent;R=A.parent
F=R/'151_FROZEN_SCENARIO_EVALUATION_20260907/runs/primary'
H=R/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2/days'
T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907'
P=R.parent/'EES_PARAGRAPH_REVIEW_MASTER_20260901'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    assert not (A/'PROTOCOL.json').exists(),'Frozen protocol already exists'
    margin=R/'166_RECORD_MARGIN_DIAGNOSTIC_20260908/RECORD_MARGINS.json'
    old=json.loads(margin.read_text('utf-8'));inputs=old['source_hashes']
    inputs[str(margin)]=sha(margin)
    for path,h in inputs.items():assert sha(Path(path))==h
    tests=json.loads((A/'UNIT_TESTS.json').read_text('utf-8'))
    assert tests['interface_sha256']==sha(A/'bounded_feedback.py')
    protocol={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'fractions':[.01,.05,.10],'signs':[-1,1],'tolerance_MW':2e-6,
              'inputs':inputs,'implementation':{n:sha(A/n) for n in ['PROTOCOL.md','bounded_feedback.py','test_bounded_feedback.py','check_fixed_pairs.py','UNIT_TESTS.json']},
              'paper_unchanged':{n:sha(P/n) for n in ['main.tex','supporting_information.tex','main.pdf','supporting_information.pdf','EES_OVERLEAF_CURRENT_20260907.zip']},
              'scope':'Fixed original allocation/replay pairs. Synthetic capacity-scaled error bounds; not calibrated telemetry, not new state trajectories or AC robustness.'}
    (A/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen',len(inputs),'source hashes; all six regimes retained')
def verify(p):
    for n,h in p['implementation'].items():assert sha(A/n)==h,n
    for path,h in p['inputs'].items():assert sha(Path(path))==h,path
    for n,h in p['paper_unchanged'].items():assert sha(P/n)==h,n
def run():
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'));verify(p);tol=p['tolerance_MW']
    with (F/'AU_EXTERNAL_INTERVAL_RESULTS.csv').open(encoding='utf-8-sig') as f:
        records={(r['date'],int(r['interval_index'])):r for r in csv.DictReader(f) if r['control']=='ECF'}
    assert len(records)==3840
    dates=sorted({d for d,t in records});assert len(dates)==80
    counters={(f,s,k):Counter() for f in p['fractions'] for s in p['signs'] for k in ['POINT','LOWER_BOUND']}
    rows=[];checked=0;maxlower=0.
    for date in dates:
        with np.load(T/'trajectories'/f'{date}.npz') as z:
            a=z['available_mw'];y=z['beta_0.5_delivery'];x=z['beta_0.5_award'];req=z['beta_0.5_request'];C=z['capacity_mw'];idle=z['beta_0.5_idle']
        with np.load(H/date/'CAPTURED_RAYS.npz') as z:
            mask=z['participant'];ids={int(t):k for k,t in enumerate(z['interval_index']) if z['control'][k]=='REPLAY'}
            xr=np.stack([z['original_factor'][ids[t]]*z['request_kw'][ids[t],mask]/1000 for t in range(48)])
            rreq=np.stack([z['request_kw'][ids[t],mask]/1000 for t in range(48)])
        assert np.max(np.abs(rreq-np.where(idle,np.minimum(req,y),req)))<1e-10
        for t in range(48):
            original=records[date,t];actual=bool(int(original['gate_triggered']))
            assert original['gate_reason']!='REPLAY_INVALID'
            base=dict(allocation=x[t],delivery=y[t],replay_allocation=xr[t],record_valid=True,replay_physical_valid=True,tolerance=tol)
            exact=assess(**base,record=a[t],error_radius=np.zeros_like(C))
            assert exact.record_valid and exact.qualifies==actual,(date,t,exact)
            assert tuple(np.flatnonzero(idle[t]))==exact.idle_indices
            checked+=1
            for fraction in p['fractions']:
                delta=fraction*C
                for sign in p['signs']:
                    record=np.maximum(y[t],a[t]+sign*delta)
                    assert np.max(np.abs(record-a[t])-delta)<1e-12
                    bounded=assess(**base,record=record,error_radius=delta)
                    point=assess(**base,record=record,error_radius=np.zeros_like(C))
                    assert bounded.record_valid and point.record_valid
                    gap=float(np.max(np.asarray(bounded.lower_availability)-a[t]));maxlower=max(maxlower,gap)
                    assert gap<1e-12
                    assert not bounded.qualifies or actual,(date,t,fraction,sign)
                    vals={}
                    for name,q in [('POINT',point),('LOWER_BOUND',bounded)]:
                        cls=('true_positive' if actual else 'false_positive') if q.qualifies else ('false_negative' if actual else 'true_negative')
                        counters[fraction,sign,name][cls]+=1
                        vals[name]={'qualifies':q.qualifies,'outside_delivery_MW':q.outside_delivery_gain,'total_delivery_MW':q.total_delivery_gain}
                    rows.append({'date':date,'interval':t,'radius_capacity_fraction':fraction,'bias_sign':sign,'true_qualification':actual,**vals})
    assert checked==3840
    summary=[]
    for (fraction,sign,name),count in counters.items():
        allcounts={k:count[k] for k in ['true_positive','false_positive','false_negative','true_negative']}
        assert sum(allcounts.values())==3840
        assert allcounts['true_positive']+allcounts['false_negative']==483
        summary.append({'radius_capacity_fraction':fraction,'bias_sign':sign,'method':name,**allcounts})
    verify(p)
    out={'protocol_sha256':sha(A/'PROTOCOL.json'),'original_pairs':checked,'exact_record_qualifying_pairs':483,
         'perturbed_pairs':len(rows),'max_lower_minus_truth_MW':maxlower,'summary':summary,'rows':rows,
         'scope':p['scope'],'new_allocation_solves':0,'new_feedback_trajectories':0,'paper_unchanged':True}
    (A/'FIXED_PAIR_RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
