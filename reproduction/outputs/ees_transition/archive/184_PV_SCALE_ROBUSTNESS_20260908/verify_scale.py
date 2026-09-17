"""Independent saved-vector and numerical-screen checks; no control changes."""
from pathlib import Path
import argparse,csv,json
from collections import Counter
import numpy as np
import run_scale as run
from audit_replay_ac import screen_predicate

A=Path(__file__).resolve().parent
CONTROL_INDEX={'NO_FEEDBACK':0,'PERSISTENCE_ONLY':1,'ECF':2,'MATCHED_UNIFORM':4}

def near(a,b,tol=1e-10):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert err<tol,(err,tol)

def inspect_day(out):
    done=run.completed(out);assert done
    result=json.loads((out/'RESULT.json').read_text('utf-8'))
    records={(r['interval_index'],r['control']):r for r in result['interval_records']}
    assert len(records)==192
    failures=[]
    with np.load(out/'VECTORS.npz') as z:
        x=z['award_mw'].reshape(48,5,-1);y=z['delivery_mw'].reshape(48,5,-1)
        a=z['available_mw'].reshape(48,5,-1);req=z['request_mw'].reshape(48,5,-1)
        load=z['load_kw'].reshape(48,5,-1);pv=z['pv_kw'].reshape(48,5,-1)
        part=z['participant'];cap=z['capacity_mw'];physical=z['physical_pair_pass'].reshape(48,5)
        for k in range(1,5):
            near(load[:,0],load[:,k]);near(pv[:,0],pv[:,k]);near(a[:,0],a[:,k])
        near(a,np.maximum(np.where(part,pv,0)-load,0)/1000)
        near(y,np.minimum(x,a))
        assert np.all(x>=0) and np.all(req>=0) and np.max(x-req)<1e-10
        for t in range(48):
            idle=x[t,2]-y[t,2]>2e-6
            assert np.array_equal(idle,z['idle'][t])
            rpl=np.where(idle,np.minimum(req[t,2],y[t,2]),req[t,2]);near(req[t,3],rpl)
            near(req[t,2],np.minimum(req[t,0],np.minimum(cap,z['ceiling_before'][t])))
            near(req[t,2].sum(),req[t,4].sum())
            if req[t,0].sum()>0:near(req[t,4],req[t,0]*req[t,2].sum()/req[t,0].sum())
            else:near(req[t,4],np.zeros_like(cap))
            for j in range(5):
                nz=req[t,j]>1e-12
                if nz.any():near(x[t,j],req[t,j]*x[t,j,nz][0]/req[t,j,nz][0])
            gains=[np.maximum(x[t,3]-x[t,2],0)[~idle].sum(),np.maximum(y[t,3]-y[t,2],0)[~idle].sum(),(y[t,3]-y[t,2]).sum()]
            near(gains,z['gains_mw'][t])
            gate=bool(idle.any() and physical[t,3] and all(g>2e-6 for g in gains))
            assert gate==bool(z['gate'][t])==bool(records[t,'ECF']['gate_triggered'])
            target=cap.copy()
            if gate:target[idle]=y[t,2,idle]
            near(target,z['ceiling_target'][t])
            near(z['ceiling_after'][t],.5*(z['ceiling_before'][t]+target))
            near(z['ceiling_before'][t],cap if t==0 else z['ceiling_after'][t-1])
            if not physical[t,3]:failures.append(dict(interval=t,control='REPLAY',kind='captured_replay_pair'))
            for control,j in CONTROL_INDEX.items():
                r=records[t,control]
                for name,value in [('authorized_export_kw',x[t,j].sum()*1000),('delivered_export_kw',y[t,j].sum()*1000),('idle_authorization_kw',(x[t,j]-y[t,j]).sum()*1000),('admitted_request_kw',req[t,j].sum()*1000)]:near(r[name],value,1e-7)
                actual=bool(r['actual_converged'] and r['actual_voltage_violation_count']==0 and r['actual_transformer_overload_count']==0)
                scalar=bool(r['actual_converged'] and .9<=r['actual_voltage_min_pu']<=r['actual_voltage_max_pu']<=1.1 and r['actual_transformer_max_loading_pu']<=1+1e-9)
                assert actual==scalar
                assert bool(r['physical_pair_pass'])==bool(actual and r['allocation_pair_pass'])==bool(physical[t,j])
                if not physical[t,j]:failures.append(dict(interval=t,control=control,kind='allocation_or_delivery',vmin=r['actual_voltage_min_pu'],vmax=r['actual_voltage_max_pu'],transformer=r['actual_transformer_max_loading_pu'],status=r['allocation_status']))
        for s in result['day_summaries']:
            j=CONTROL_INDEX[s['control']]
            near(s['delivered_export_mwh'],y[:,j].sum()*.5)
            near(s['authorized_export_mwh'],x[:,j].sum()*.5)
            near(s['idle_authorization_mwh'],(x[:,j]-y[:,j]).sum()*.5)
    return result,failures

def main(partial):
    p=run.verify_sources();rows=[];failures=[];files={};count=0;ac_days=0;contradictions=[];exposure={}
    from scale_inputs import InputReference
    reference=InputReference(p)
    for m in p['settings']:
        capacity=float(reference.expected(m['factor'],p['dates'][0])['capacity'].sum())
        exposure[m['id']]=dict(participant_capacity_MW=capacity,no_feedback_request_limited_intervals=0,
            no_feedback_request_limited_physical_pass_intervals=0,available_export_MWh=0.)
        for date in p['dates']:
            out=A/'days'/m['id']/date
            if not (out/'DONE.json').exists():continue
            result,bad=inspect_day(out);count+=1
            reference.verify(out,m['factor'],date)
            with np.load(out/'VECTORS.npz') as z:
                requests=z['request_mw'].reshape(48,5,-1)[:,0]
                awards=z['award_mw'].reshape(48,5,-1)[:,0]
                limited=(requests-awards).sum(axis=1)>2e-6
                good=z['physical_pair_pass'].reshape(48,5)[:,0]
                exposure[m['id']]['no_feedback_request_limited_intervals']+=int(limited.sum())
                exposure[m['id']]['no_feedback_request_limited_physical_pass_intervals']+=int((limited & good).sum())
                exposure[m['id']]['available_export_MWh']+=float(z['available_mw'].reshape(48,5,-1)[:,0].sum()*.5)
            failures.extend(dict(setting=m['id'],date=date,**b) for b in bad)
            files[m['id']+'/'+date]=run.completed(out)['files']
            if (out/'REPLAY_AC.json').exists():
                audit=json.loads((out/'REPLAY_AC.json').read_text('utf-8'))
                assert audit['controller_files']==files[m['id']+'/'+date]
                assert audit['protocol_sha256']==run.sha(A/'PROTOCOL.json')
                assert audit['audit_sha256']==run.sha(A/'audit_replay_ac.py')
                assert len(audit['screens'])==48
                for t,s in enumerate(audit['screens']):
                    assert t==s['interval']
                    bp=screen_predicate(s['load_only']);up=screen_predicate(s['authorization']);sp=screen_predicate(s['delivery'])
                    good=bool(bp and up and sp and s['authorization']['customer_voltage_max_pu']<=1.07+1e-9)
                    assert good==s['recomputed_pair_pass']
                    if good!=s['recorded_pair_pass']:contradictions.append(dict(setting=m['id'],date=date,interval=t))
                    if not good:failures.append(dict(setting=m['id'],date=date,interval=t,control='REPLAY',kind='post_decision_AC',used_for_tightening=s['used_for_tightening']))
                assert len(audit['contradictions'])==sum(s['recorded_pair_pass']!=s['recomputed_pair_pass'] for s in audit['screens'])
                files[m['id']+'/'+date]=dict(files[m['id']+'/'+date],REPLAY_AC=run.sha(out/'REPLAY_AC.json'))
                ac_days+=1
            for s in result['day_summaries']:
                records=[r for r in result['interval_records'] if r['control']==s['control']]
                rows.append(dict(setting=m['id'],**s,source_terminal_MWh=sum(r['actual_source_terminal_p_kw'] for r in records)*.5/1000))
    report=dict(complete=count==p['expected_new_days'] and ac_days==p['expected_new_days'] and not contradictions,verified_days=count,control_intervals=192*count,replay_intervals=48*count,post_decision_AC_replays=48*ac_days,failures=failures,contradictions=contradictions,protocol_sha256=run.sha(A/'PROTOCOL.json'),verifier_sha256=run.sha(Path(__file__)),scaled_inputs_and_requests_verified=True,scope='Saved vector identities, numerical predicates and separately executed replay AC checks using the frozen solver.')
    if not report['complete']:
        assert partial,'All scale-days are required'
        (A/'PARTIAL_VECTOR_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report));return
    dates=np.array(p['dates'],dtype='datetime64[D]').astype(int)
    kernel=np.maximum(1-np.abs(dates[:,None]-dates[None,:])/8,0)
    eig,v=np.linalg.eigh(kernel)
    weights=np.random.default_rng(184).standard_normal((5000,80))@(v*np.sqrt(np.maximum(eig,0))).T
    weights=np.concatenate([weights,-weights]);index={(r['setting'],r['date'],r['control']):r for r in rows};contrasts={};totals={}
    for m in p['settings']:
        name=m['id'];totals[name]={}
        for control in CONTROL_INDEX:
            selected=[index[name,d,control] for d in p['dates']]
            totals[name][control]={k:sum(r[k] for r in selected) for k in ['delivered_export_mwh','authorized_export_mwh','idle_authorization_mwh','gate_count','source_terminal_MWh','power_flow_count']}
        ecf=np.array([index[name,d,'ECF']['delivered_export_mwh'] for d in p['dates']])
        for control in ['NO_FEEDBACK','MATCHED_UNIFORM','PERSISTENCE_ONLY']:
            other=np.array([index[name,d,control]['delivered_export_mwh'] for d in p['dates']])
            diff=ecf-other;mean=diff.mean();ci=np.quantile(mean+weights@(diff-mean)/80,[.025,.975])
            contrasts[name+'_minus_'+control]=dict(total_MWh=float(diff.sum()),mean_daily_MWh=float(mean),date_dependent_95_MWh=ci.tolist(),positive_days=int((diff>1e-10).sum()),negative_days=int((diff<-1e-10).sum()),zero_days=int((np.abs(diff)<=1e-10).sum()))
            contrasts[name+'_minus_'+control]['MWh_per_installed_participant_MW']=float(diff.sum())/exposure[name]['participant_capacity_MW']
        missing=exposure[name]['available_export_MWh']-totals[name]['NO_FEEDBACK']['delivered_export_mwh']
        assert missing>=-1e-10
        exposure[name]['no_feedback_undelivered_available_MWh']=missing
        exposure[name]['recovered_fraction_of_no_feedback_undelivered_available']=contrasts[name+'_minus_NO_FEEDBACK']['total_MWh']/missing if missing>1e-10 else None
    report.update(totals=totals,contrasts=contrasts,daily=rows,output_hashes=files,exposure=exposure)
    run.verify_sources()
    (A/'VERIFIED_SCALE_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['daily','output_hashes']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();main(args.partial)
