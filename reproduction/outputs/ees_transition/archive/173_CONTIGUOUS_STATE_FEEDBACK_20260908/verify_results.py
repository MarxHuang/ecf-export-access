"""Independent vector, calendar and energy checks; no partial headline estimates."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,math
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent;F=R/'151_FROZEN_SCENARIO_EVALUATION_20260907';B=R/'169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(partial):
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'));ph=sha(A/'PROTOCOL.json')
    assert sha(A/'run_contiguous.py')==p['implementation_sha256']
    assert sha(A/'PROTOCOL.md')==p['notes_sha256']
    counts=Counter();errors=Counter();rows=[];flags=[];blockstatus=[];bounds=[];outputs={}
    dailyrefs={}
    for beta,case in [(0.25,'beta_025_exact'),(.5,'primary'),(1.,'beta_100_exact')]:
        with (F/'runs'/case/'AU_EXTERNAL_DAY_SUMMARY.csv').open(encoding='utf-8-sig') as f:
            dailyrefs[beta]={(r['date'],r['control']):float(r['delivered_export_mwh']) for r in csv.DictReader(f)}
    metrics={'delivery':'delivered_export_kw','authorization':'authorized_export_kw','unused':'idle_authorization_kw','source':'actual_source_terminal_p_kw','loss':'actual_circuit_loss_kw'}
    for job in p['jobs']:
        beta=job['beta'];ecf=job['policy']=='ECF';control='ECF' if ecf else 'UNQUALIFIED_SAME_UPDATE';prev=None;completed=0
        for di,d in enumerate(job['dates']):
            out=A/'jobs'/job['id']/d
            if not (out/'DONE.json').exists():break
            done=json.loads((out/'DONE.json').read_text('utf-8'))
            assert done['protocol_sha256']==ph
            for n,h in done['files'].items():assert sha(out/n)==h
            outputs[job['id']+'/'+d]=done['files']
            j=json.loads((out/'RESULT.json').read_text('utf-8'));rec={(q['control'],q['interval_index']):q for q in j['records']}
            assert len(rec)==(96 if ecf else 48)
            with np.load(out/'TRAJECTORIES.npz') as z:
                mask=z['participant'];cap=z['capacity_full_mw'][mask]
                state=cap.copy() if prev is None else prev.copy()
                assert np.max(np.abs(state-z['start_ceiling_full_mw'][mask]))<1e-12
                start_deficit=float(np.max(cap-state))*1000
                if di:counts['verified_day_boundaries']+=1
                raw=z['raw_request_mw'];x=z['award'];y=z['delivery'];a=z['available'];ceil=z['ceiling'];target=z['target'];req=z['request'];idle=z['idle']
                assert np.all(x>=-1e-12) and np.all(y>=-1e-12)
                assert np.max(y-x)<1e-10 and np.max(x-req)<1e-10
                assert np.max(np.abs(y-np.minimum(x,a)))<1e-10
                for t in range(48):
                    q=rec[control,t];assert np.max(np.abs(ceil[t]-state))<1e-12
                    assert np.max(np.abs(req[t]-np.minimum(raw[t],np.minimum(cap,state))))<1e-12
                    assert np.array_equal(idle[t],x[t]-y[t]>2e-6)
                    enabled=bool(q['gate_triggered']) if ecf else bool(idle[t].any())
                    assert enabled==q['delivery_target_enabled']
                    expected=cap.copy()
                    if enabled:expected[idle[t]]=y[t,idle[t]]
                    assert np.max(np.abs(expected-target[t]))<1e-12
                    after=(1-beta)*state+beta*expected
                    assert np.all(after>=-1e-12) and np.all(after<=cap+1e-12)
                    for key,arr in [('admitted_request_kw',req),('authorized_export_kw',x),('delivered_export_kw',y)]:
                        e=abs(math.fsum(arr[t])*1000-q[key]);errors[key]=max(errors[key],e);assert e<1e-7
                    assert abs(q['authorized_export_kw']-q['delivered_export_kw']-q['idle_authorization_kw'])<1e-7
                    if ecf:
                        rx=z['replay_award'][t];ry=z['replay_delivery'][t]
                        assert np.max(np.abs(ry-np.minimum(rx,a[t])))<1e-10
                        gains=[float(np.maximum(rx[~idle[t]]-x[t,~idle[t]],0).sum()),float(np.maximum(ry[~idle[t]]-y[t,~idle[t]],0).sum()),float(math.fsum(ry)-math.fsum(y[t]))]
                        for key,g in zip(['outsider_access_gain_kw','outsider_delivery_gain_kw','replay_delivery_gain_kw'],gains):assert abs(q[key]-g*1000)<1e-7
                        qualified=bool(idle[t].any() and q['replay_physical_pass'] and min(gains)>2e-6)
                        assert qualified==bool(q['gate_triggered'])
                        ur=z['uniform_request'][t];ux=z['uniform_award'][t];uy=z['uniform_delivery'][t]
                        factor=req[t].sum()/raw[t].sum() if raw[t].sum() else 1.
                        assert np.max(np.abs(ur-raw[t]*factor))<1e-10
                        assert abs(ur.sum()-req[t].sum())<1e-10
                        assert np.max(np.abs(uy-np.minimum(ux,a[t])))<1e-10
                        counts['replays_checked']+=1
                        if not q['replay_physical_pass']:flags.append({'job':job['id'],'date':d,'interval':t,'kind':'replay'})
                    counts['state_updates_checked']+=1
                    counts['ceiling_reductions']+=int(np.sum(after<state-1e-12))
                    state=after
                assert np.max(np.abs(state-z['end_ceiling_full_mw'][mask]))<1e-12
                prev=state.copy()
                for c in ([control,'MATCHED_UNIFORM'] if ecf else [control]):
                    rr=[rec[c,t] for t in range(48)]
                    row={'job':job['id'],'block':job['block'],'date':d,'beta':beta,'control':c}
                    row.update({k:math.fsum(q[col] for q in rr)*.5/1000 for k,col in metrics.items()})
                    row['qualifying_intervals']=sum(q['gate_triggered'] for q in rr);row['power_flow_calls']=sum(q['power_flow_count'] for q in rr)
                    assert abs(row['authorization']-row['delivery']-row['unused'])<1e-10
                    if c=='ECF':row['difference_from_daily_reset_MWh']=row['delivery']-dailyrefs[beta][d,'ECF']
                    for q in rr:
                        actual=bool(q['actual_converged'] and q['actual_voltage_violation_count']==0 and q['actual_transformer_overload_count']==0)
                        assert actual==bool(q['actual_converged'] and .90<=q['actual_voltage_min_pu']<=q['actual_voltage_max_pu']<=1.10 and q['actual_transformer_max_loading_pu']<=1.+1e-9)
                        if q['allocation_pair_pass']:assert q['load_only_pair_pass'] and q['authorized_converged'] and q['authorized_voltage_min_pu']>=.9 and q['authorized_voltage_max_pu']<=1.07+1e-9 and q['authorized_transformer_max_loading_pu']<=1.+1e-9
                        assert bool(q['physical_pair_pass'])==bool(actual and q['allocation_pair_pass'])
                        if not q['physical_pair_pass']:flags.append({'job':job['id'],'date':d,'control':c,'interval':q['interval_index'],'kind':'observed'})
                        counts['observed_records_checked']+=1
                    rows.append(row)
                bounds.append({'job':job['id'],'date':d,'day_in_block':di+1,'max_start_deficit_kW':start_deficit,'max_end_deficit_kW':float(np.max(cap-state))*1000})
            completed+=1
        blockstatus.append({'job':job['id'],'completed':completed,'expected':len(job['dates'])})
    complete=all(x['completed']==x['expected'] for x in blockstatus)
    result={'protocol_sha256':ph,'complete':complete,'counts':dict(counts),'jobs':blockstatus,'physical_failures':flags,'max_errors':dict(errors),'verification_scope':'Independent ceiling recursion, paired energy and observed-screen fields; replay AC pass flags are original solver outputs, replay delivery gains independently recomputed.'}
    if not complete:
        assert partial,'Full experiment not yet complete'
        (A/'PROGRESS_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ['jobs']},ensure_ascii=True));return
    dates=p['dates'];t=np.array(dates,dtype='datetime64[D]').astype(int);K=np.maximum(1-np.abs(t[:,None]-t[None,:])/8,0.)
    eig,V=np.linalg.eigh(K);assert eig.min()>-1e-10
    Z=np.random.default_rng(173).standard_normal((5000,80))@(V*np.sqrt(np.maximum(eig,0))).T;W=np.concatenate([Z,-Z])
    index={(r['beta'],r['control'],r['date']):r for r in rows};contrasts={}
    for beta in [.25,.5,1.]:
        y=np.array([index[beta,'ECF',d]['delivery'] for d in dates])
        references={'NO_FEEDBACK':np.array([dailyrefs[beta][d,'NO_FEEDBACK'] for d in dates]),'MATCHED_UNIFORM':np.array([index[beta,'MATCHED_UNIFORM',d]['delivery'] for d in dates]),'DAILY_RESET_ECF':np.array([dailyrefs[beta][d,'ECF'] for d in dates])}
        if beta==.5:references['UNQUALIFIED_SAME_UPDATE']=np.array([index[.5,'UNQUALIFIED_SAME_UPDATE',d]['delivery'] for d in dates])
        for name,ref in references.items():
            diff=y-ref;mean=float(diff.mean());ci=np.quantile(mean+W@(diff-mean)/80,[.025,.975])
            blocktotals=[float(sum(diff[dates.index(d)] for d in ds)) for ds in p['blocks']]
            contrasts[f'beta_{beta}_minus_{name}']={'total_delivery_MWh':float(y.sum()),'difference_MWh':float(diff.sum()),'mean_difference_MWh_per_day':mean,'date_dependent_wild_95':ci.tolist(),'positive_days':int((diff>1e-10).sum()),'negative_days':int((diff<-1e-10).sum()),'zero_days':int((abs(diff)<=1e-10).sum()),'block_totals_MWh':blocktotals,'daily_difference_MWh':diff.tolist()}
    for s,h in p['inputs'].items():assert sha(Path(s))==h,s
    result.update(contrasts=contrasts,daily=rows,boundary_deficits=bounds,output_hashes=outputs)
    (A/'VERIFIED_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['daily','boundary_deficits','output_hashes','jobs']},ensure_ascii=True,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();main(args.partial)
