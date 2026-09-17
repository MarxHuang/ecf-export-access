"""Independent post-run trace/record verification and pre-specified paired analysis."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,csv
import numpy as np
A=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'));plan=json.loads((A/'ANALYSIS_PLAN.json').read_text('utf-8'))
    assert all(sha(Path(s))==h for s,h in p['inputs'].items())
    assert sha(A/'run_comparison.py')==p['implementation_sha256']
    dates=p['dates'];ctrls=p['controls'];rows=[];members={c:np.zeros(418) for c in ctrls};tight={c:np.zeros(418,dtype=int) for c in ctrls};counts=Counter();errs=Counter();dayhash={}
    flags=[];metrics={'delivery':'delivered_export_kw','authorization':'authorized_export_kw','unused':'idle_authorization_kw','source':'actual_source_terminal_p_kw','loss':'actual_circuit_loss_kw','available_not_delivered':'curtailed_available_export_kw'}
    for date in dates:
        folder=A/'days'/date;done=json.loads((folder/'DONE.json').read_text('utf-8'))
        assert done['protocol_sha256']==sha(A/'PROTOCOL.json')
        assert all(sha(folder/n)==h for n,h in done['files'].items())
        dayhash[date]=done['files']
        j=json.loads((folder/'RESULT.json').read_text('utf-8'))
        record={(r['control'],r['interval_index']):r for r in j['records']};assert len(record)==96
        d={'date':date}
        with np.load(folder/'TRAJECTORIES.npz') as z:
            cap=z['capacity_mw'];raw=z['raw_request_mw'];assert cap.shape==(418,)
            for c in ctrls:
                before=cap.copy();x=z[c+'_award'];y=z[c+'_delivery'];r=z[c+'_request'];ceil=z[c+'_ceiling'];target=z[c+'_target'];idle=z[c+'_idle']
                assert np.all(x>=0) and np.all(y>=0) and np.all(y<=x+1e-12) and np.all(x<=r+1e-12)
                for t in range(48):
                    q=record[c,t];assert np.max(np.abs(ceil[t]-before))<1e-12
                    assert np.max(np.abs(r[t]-np.minimum(raw[t],np.minimum(cap,before))))<1e-12
                    assert np.array_equal(idle[t],x[t]-y[t]>2e-6)
                    enabled=bool(q['replay_qualified']) if c=='ECF' else bool(idle[t].any())
                    expected=cap.copy()
                    if enabled:expected[idle[t]]=y[t,idle[t]]
                    assert np.max(np.abs(expected-target[t]))<1e-12
                    after=.5*before+.5*expected
                    assert q['tightened_members']==int(np.sum(after<before-1e-12))
                    tight[c]+=(after<before-1e-12)
                    for key,arr in [('admitted_request_kw',r),('authorized_export_kw',x),('delivered_export_kw',y)]:
                        e=abs(math.fsum(map(float,arr[t]))*1000-q[key]);errs[key]=max(errs[key],e);assert e<1e-7
                    assert abs(q['authorized_export_kw']-q['delivered_export_kw']-q['idle_authorization_kw'])<1e-7
                    if c!='ECF':assert q['replay_qualified'] is None and q['replay_power_flow_count']==0
                    actual_ok=bool(q['actual_converged'] and q['actual_voltage_violation_count']==0 and q['actual_transformer_overload_count']==0)
                    assert actual_ok==bool(q['actual_converged'] and .90<=q['actual_voltage_min_pu']<=q['actual_voltage_max_pu']<=1.10 and q['actual_transformer_max_loading_pu']<=1.0+1e-9)
                    if q['allocation_pair_pass']:
                        assert q['load_only_pair_pass'] and q['authorized_converged'] and q['authorized_voltage_min_pu']>=.9 and q['authorized_voltage_max_pu']<=1.07+1e-9 and q['authorized_transformer_max_loading_pu']<=1.0+1e-9
                    assert bool(q['physical_pair_pass'])==bool(q['allocation_pair_pass'] and actual_ok)
                    assert abs(q['available_export_kw']-record[ctrls[0],t]['available_export_kw'])<1e-7
                    if not q['physical_pair_pass']:flags.append({'date':date,'control':c,'interval':t,'voltage_min':q['actual_voltage_min_pu'],'voltage_max':q['actual_voltage_max_pu'],'transformer':q['actual_transformer_max_loading_pu'],'status':q['allocation_status']})
                    counts[c+'_physical_pass']+=int(q['physical_pair_pass']);counts[c+'_power_flow_calls']+=q['power_flow_count'];counts[c+'_tightening_updates']+=q['tightened_members'];counts[c+'_update_intervals']+=int(enabled and idle[t].any())
                    before=after
                members[c]+=y.sum(axis=0)*.5
                for metric,col in metrics.items():d[c+'_'+metric]=math.fsum(record[c,t][col] for t in range(48))*.5/1000
                assert abs(d[c+'_authorization']-d[c+'_delivery']-d[c+'_unused'])<1e-10
        for metric in metrics:d['difference_'+metric]=d['ECF_'+metric]-d['UNQUALIFIED_SAME_UPDATE_'+metric]
        rows.append(d)
    days=np.array(dates,dtype='datetime64[D]').astype(int);K=np.maximum(1-np.abs(days[:,None]-days[None,:])/8,0.)
    eig,V=np.linalg.eigh(K);assert eig.min()>-1e-10
    Z=np.random.default_rng(170).standard_normal((5000,len(dates)))@(V*np.sqrt(np.maximum(eig,0))).T
    weights=np.concatenate([Z,-Z]);indices=np.random.default_rng(169).integers(0,80,size=(10000,80))
    paired={}
    for metric in metrics:
        values=np.array([r['difference_'+metric] for r in rows]);mean=float(values.mean())
        percentile=np.quantile(values[indices].mean(axis=1),[.025,.975])
        wild=np.quantile(mean+weights@(values-mean)/80,[.025,.975])
        paired[metric]={'sum_MWh':float(values.sum()),'mean_MWh_per_day':mean,'paired_percentile_95':percentile.tolist(),'date_dependent_wild_95':wild.tolist(),'positive_days':int((values>1e-10).sum()),'negative_days':int((values<-1e-10).sum()),'zero_days':int((abs(values)<=1e-10).sum())}
    diff=members['ECF']-members['UNQUALIFIED_SAME_UPDATE']
    individual={'positive':int((diff>1e-10).sum()),'negative':int((diff<-1e-10).sum()),'zero':int((abs(diff)<=1e-10).sum()),'difference_MWh_quantiles':np.quantile(diff,[0,.05,.25,.5,.75,.95,1]).tolist(),'difference_MWh_by_connection_index':diff.tolist(),'tightening_updates_by_connection':{c:v.tolist() for c,v in tight.items()}}
    assert abs(diff.sum()-paired['delivery']['sum_MWh'])<1e-9
    totals={c:{metric:math.fsum(r[c+'_'+metric] for r in rows) for metric in metrics} for c in ctrls}
    result={'protocol_sha256':sha(A/'PROTOCOL.json'),'analysis_plan_sha256':sha(A/'ANALYSIS_PLAN.json'),'days':80,'controls':ctrls,'counts':dict(counts),'totals':totals,'paired':paired,'individual':individual,'physical_failures':flags,'max_errors':dict(errs),'daily':rows,'output_hashes':dayhash}
    assert all(sha(Path(s))==h for s,h in p['inputs'].items())
    (A/'VERIFIED_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['individual','daily','output_hashes']},indent=2))
if __name__=='__main__':main()
