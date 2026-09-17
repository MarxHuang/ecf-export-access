"""Reconstruct saved learning and ceiling updates without the simulation policy."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from run_adaptation import A,BRANCHES,verify,sha,read,write

def close(x,y,tol=1e-11):
    x=np.asarray(x);y=np.asarray(y)
    assert x.shape==y.shape and np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    err=float(np.max(np.abs(x-y))) if x.size else 0.
    assert err<=tol,err

def independent_draw(seed,d,t,n):
    digest=hashlib.sha256(f'ECF_ADAPTATION_189|{seed}|{d}|{t}'.encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16],'big'))).random(n)

def audit(partial=False,smoke=False):
    p=verify();ph=sha(A/'PROTOCOL.json');rows=[];files={};bad=[];updates=0;qualified=0;completed=0;replays=0;changed=0
    if smoke:jobs=[{'id':'FIXED_REFERENCE','mode':'FIXED_REFERENCE','seed':2001,'block':1,'dates':[p['dates'][0]]}]
    else:jobs=p['jobs']
    for job in jobs:
        previous=None
        for di,d in enumerate(job['dates']):
            out=A/('smoke' if smoke else 'jobs')/job['id']/d
            if not (out/'DONE.json').exists():
                assert partial,(job['id'],d,'missing')
                break
            done=read(out/'DONE.json');assert done['protocol_sha256']==ph
            for n,h in done['files'].items():assert sha(out/n)==h;files[str(out/n)]=h
            result=read(out/'RESULT.json');assert result['job']==job and result['protocol_sha256']==ph
            records=result['records'];assert len(records)==192
            ix={(q['control'],int(q['interval_index'])):q for q in records};assert len(ix)==192
            with np.load(out/'TRAJECTORIES.npz') as z:
                cap=z['capacity_mw'];n=len(cap);assert n==418
                base=z['base_request_mw'];assert base.shape==(48,n)
                for c in BRANCHES:
                    state=z[c+'_start_ceiling'].copy();weights=z[c+'_start_weights'].copy()
                    if previous is None:close(state,cap);close(weights,np.zeros((n,5)))
                    else:close(state,previous[c][0]);close(weights,previous[c][1])
                    for t in range(48):
                        q=ix[c,t];u=independent_draw(job['seed'],d,t,n)
                        w=np.exp(weights-weights.max(axis=1,keepdims=True));probs=.9*w/w.sum(axis=1,keepdims=True)+.1/5
                        close(probs,z[c+'_probabilities'][t])
                        cdf=probs.cumsum(axis=1);cdf[:,-1]=1
                        actions=(u[:,None]>=cdf).sum(axis=1)
                        if job['mode']=='FIXED_REFERENCE':actions[:]=2
                        assert np.array_equal(actions,z[c+'_action'][t])
                        raw=np.minimum(base[t]*np.array([.5,.75,1.,1.25,1.5])[actions],cap)
                        req=raw if c=='NO_FEEDBACK' else np.minimum(raw,state)
                        close(raw,z[c+'_raw'][t]);close(req,z[c+'_request'][t]);close(state,z[c+'_ceiling'][t])
                        y=z[c+'_delivery'][t];x=z[c+'_award'][t];avail=z[c+'_available'][t]
                        assert np.all(x>=-1e-12) and np.all(x<=req+1e-10)
                        close(y,np.minimum(x,avail));idle=x-y>2e-6
                        assert np.array_equal(idle,z[c+'_idle'][t])
                        if c=='ECF':
                            rr=req.copy();rr[idle]=np.minimum(req[idle],y[idle]);close(rr,z['REPLAY_request'][t])
                            rx=z['REPLAY_award'][t];ry=z['REPLAY_delivery'][t]
                            close(ry,np.minimum(rx,avail));assert np.all(rx<=rr+1e-10)
                            ag=np.maximum(rx[~idle]-x[~idle],0).sum();dg=np.maximum(ry[~idle]-y[~idle],0).sum();tg=ry.sum()-y.sum()
                            enabled=bool(idle.any() and q['replay_physical_pass'] and ag>2e-6 and dg>2e-6 and tg>2e-6)
                            assert enabled==bool(q['gate_triggered'])
                            for k,v in [('outsider_access_gain_kw',ag*1000),('outsider_delivery_gain_kw',dg*1000),('replay_delivery_gain_kw',tg*1000)]:close(np.array(q[k]),np.array(v),1e-7)
                            qualified+=int(enabled);replays+=1
                            uq=ix['MATCHED_UNIFORM_SHADOW',t];ur=z['MATCHED_UNIFORM_SHADOW_request'][t]
                            close(ur,raw*float(uq['uniform_factor']));close(np.array(ur.sum()),np.array(req.sum()))
                            ux=z['MATCHED_UNIFORM_SHADOW_award'][t];uy=z['MATCHED_UNIFORM_SHADOW_delivery'][t]
                            close(uy,np.minimum(ux,avail));assert np.all(ux<=ur+1e-10)
                            for k,v in [('authorized_export_kw',ux.sum()*1000),('delivered_export_kw',uy.sum()*1000)]:close(np.array(uq[k]),np.array(v),1e-7)
                        else:enabled=bool(idle.any()) if c=='UNQUALIFIED_SAME_UPDATE' else False
                        assert enabled==bool(q['delivery_target_enabled'])
                        target=cap.copy()
                        if enabled:target[idle]=y[idle]
                        close(target,z[c+'_target'][t]);state=cap.copy() if c=='NO_FEEDBACK' else .5*state+.5*target
                        reward=y/cap;close(reward,z[c+'_reward'][t])
                        if job['mode']=='ADAPTIVE':
                            eligible=np.flatnonzero(base[t]>0)
                            weights[eligible,actions[eligible]]+=.1*reward[eligible]/(5*probs[eligible,actions[eligible]])
                            weights-=weights.max(axis=1,keepdims=True)
                        for k,v in [('raw_request_kw',raw.sum()*1000),('admitted_request_kw',req.sum()*1000),('authorized_export_kw',x.sum()*1000),('delivered_export_kw',y.sum()*1000),('idle_authorization_kw',(x-y).sum()*1000)]:close(np.array(q[k]),np.array(v),1e-7)
                        changed+=int(np.sum(np.abs(req-(base[t] if c=='NO_FEEDBACK' else np.minimum(base[t],z[c+'_ceiling'][t])))>2e-6))
                        updates+=n
                    close(weights,z[c+'_end_weights']);close(state,z[c+'_end_ceiling'])
                previous={c:(z[c+'_end_ceiling'].copy(),z[c+'_end_weights'].copy()) for c in BRANCHES}
            for q in records:
                actual=bool(q['actual_converged'] and q['actual_voltage_violation_count']==0 and q['actual_transformer_overload_count']==0)
                combined=bool(q['allocation_pair_pass'] and actual)
                assert combined==bool(q['physical_pair_pass'])
                if not combined or q['replay_physical_pass'] is False:bad.append({'job':job['id'],'date':d,'interval':q['interval_index'],'control':q['control'],'observed_pass':combined,'replay_pass':q['replay_physical_pass']})
            for c in BRANCHES+['MATCHED_UNIFORM_SHADOW']:
                qs=[ix[c,t] for t in range(48)]
                row={k:job[k] for k in ['mode','seed','block']};row.update(date=d,control=c)
                for field,label in [('authorized_export_kw','authorization_mwh'),('delivered_export_kw','delivery_mwh'),('idle_authorization_kw','idle_mwh'),('actual_source_terminal_p_kw','source_exchange_mwh')]:row[label]=sum(float(q[field]) for q in qs)*.5/1000
                row['qualified_intervals']=sum(int(q['gate_triggered']) for q in qs)
                rows.append(row)
            completed+=1
    if not partial:assert completed==(1 if smoke else 480)
    summary={'complete':not partial,'smoke':smoke,'verified_setting_days':completed,'planned_setting_days':1 if smoke else 480,'participant_updates_checked':updates,'changed_admitted_participant_intervals':changed,'replays_checked':replays,'qualified_replays':qualified,'physical_failures':bad,'protocol_sha256':ph,'auditor_sha256':sha(Path(__file__)),'files':files,'scope':'Independent saved-trajectory algebra, learning, information chronology, continuity, and recorded screen-status checks; not an independent AC solver rerun.'}
    name='SMOKE_AUDIT' if smoke else 'PARTIAL_AUDIT' if partial else 'VERIFIED_ADAPTATION'
    write(A/(name+'.json'),summary)
    if rows:
        with (A/(name+'_DAILY.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['files','physical_failures']}|{'physical_failure_count':len(bad)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();audit(a.partial,a.smoke)
