"""Carry request-ceiling state only across genuinely consecutive observed days."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,time
from datetime import date,timedelta
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent
B=R/'169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908'
F=R/'151_FROZEN_SCENARIO_EVALUATION_20260907'
H=R/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2'
T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907'
PROTOCOL=A/'PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load_runner():
    s=importlib.util.spec_from_file_location('original_au_runner',F/'scientific_source/tools/run_ees_australian_external_replay.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def freeze():
    assert not PROTOCOL.exists(),'Never overwrite frozen protocol'
    old=json.loads((B/'PROTOCOL.json').read_text('utf-8'))
    days=old['dates'];blocks=[]
    for d in days:
        if not blocks or date.fromisoformat(d)-date.fromisoformat(blocks[-1][-1])!=timedelta(days=1):blocks.append([])
        blocks[-1].append(d)
    assert len(days)==80 and len(set(days))==80
    paths=dict(old['inputs'])
    for n in ['PROTOCOL.json','run_comparison.py','VERIFIED_COMPARISON.json']:
        paths[str(B/n)]=sha(B/n)
    for d in days:
        for n in ['TRAJECTORIES.npz','RESULT.json']:
            file=B/'days'/d/n
            check=json.loads((file.parent/'DONE.json').read_text('utf-8'))
            paths[str(file)]=check['files'][n]
    source=json.loads((F/'VERIFIED_RESULTS.json').read_text('utf-8'))
    for case in ['primary','beta_025_exact','beta_100_exact']:
        for n,h in source['results'][case]['input_hashes'].items():paths[str(F/'runs'/case/n)]=h
    for s,h in paths.items():assert sha(Path(s))==h,s
    jobs=[]
    for i,ds in enumerate(blocks):
        for policy,beta in [('ECF',.25),('ECF',.5),('ECF',1.),('UNQUALIFIED',.5)]:
            jobs.append({'id':f'block{i+1:02}_{policy}_{beta:g}','block':i+1,'policy':policy,'beta':beta,'dates':ds})
    p={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dates':days,'blocks':blocks,'jobs':jobs,'config':old['config'],
       'intervention':'Retain request ceiling across consecutive observed dates; reset only at starts of calendar-contiguous blocks.',
       'original_daily_reset_comparators':{'0.25':'beta_025_exact','0.5':'primary','1.0':'beta_100_exact'},
       'reset_physical_model_daily':True,'numerical_tolerances':{'power_MW':1e-10,'update_MW':1e-12,'first_day_power_kW':1e-7,'source_loss_kW':1e-4},
       'uncertainty':{'replicates':10000,'seed':173,'kernel':'max(1 - absolute calendar gap / 8, 0)','antithetic':True,'block_totals':'report all 10 blocks separately, no independent interval/participant confidence claims'},
       'inputs':paths,'implementation_sha256':sha(Path(__file__)),'notes_sha256':sha(A/'PROTOCOL.md')}
    PROTOCOL.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'blocks':blocks,'jobs':len(jobs),'input_files':len(paths),'protocol_sha256':sha(PROTOCOL)}))
def verify():
    p=json.loads(PROTOCOL.read_text('utf-8'))
    assert sha(Path(__file__))==p['implementation_sha256'] and sha(A/'PROTOCOL.md')==p['notes_sha256']
    for s,h in p['inputs'].items():assert sha(Path(s))==h,s
    return p
def tests():
    p=verify();m=load_runner();rng=np.random.default_rng(173);n=0
    for beta in [.25,.5,1.]:
        for k in range(100):
            cap=rng.uniform(.001,.1,30);state=rng.uniform(0,1,30)*cap
            initial=state.copy()
            for t in range(48*3):
                closed=t%11!=0;y=rng.uniform(0,1,30)*state;idle=tuple(np.flatnonzero(state-y>2e-6))
                expected=cap.copy()
                if not closed:expected[list(idle)]=y[list(idle)]
                after,target=m.update_access_ceiling(ceiling_before_mw=state,registered_capacity_mw=cap,delivery_mw=y,idle_indices=idle,gate_triggered=not closed,beta=beta)
                assert np.max(np.abs(np.asarray(target)-expected))<1e-15
                assert np.max(np.abs(np.asarray(after)-((1-beta)*state+beta*expected)))<1e-15
                assert np.all(np.asarray(after)>=0) and np.all(np.asarray(after)<=cap+1e-15)
                state=np.asarray(after);n+=1
            state=initial.copy()
            for t in range(20):
                state=np.asarray(m.update_access_ceiling(ceiling_before_mw=state,registered_capacity_mw=cap,delivery_mw=state,idle_indices=(),gate_triggered=False,beta=beta)[0])
            assert np.max(np.abs((cap-state)-(1-beta)**20*(cap-initial)))<1e-15
    assert sum(map(len,p['blocks']))==80 and max(map(len,p['blocks']))==35
    out={'tested_updates':n,'closed_condition_recovery_tests':300,'blocks':len(p['blocks']),'protocol_sha256':sha(PROTOCOL)}
    (A/'UNIT_TESTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
def job_task(job):
    p=json.loads(PROTOCOL.read_text('utf-8'));cfg=p['config'];m=load_runner();beta=job['beta'];is_ecf=job['policy']=='ECF'
    connections=m.parse_dss_load_connections(cfg['dss_root'])
    profile,part=m._load_mapping(Path(cfg['mapping_file']),connections)
    with np.load(Path(cfg['package'])/'AUSGRID_EVALUATION_2012_2013.npz') as z:
        dateindex={str(d):i for i,d in enumerate(z['dates'])}
        gross_data={d:m.PV_SCALE*z['gross_generation_kw'][dateindex[d],:,:].astype(float)[:,profile] for d in job['dates']}
    root=A/'jobs'/job['id'];root.mkdir(parents=True,exist_ok=True);state=None;cap_ref=None
    control='ECF' if is_ecf else 'UNQUALIFIED_SAME_UPDATE'
    for di,d in enumerate(job['dates']):
        out=root/d;out.mkdir(exist_ok=True)
        if (out/'DONE.json').exists():
            done=json.loads((out/'DONE.json').read_text('utf-8'))
            assert done['protocol_sha256']==sha(PROTOCOL)
            assert all(sha(out/n)==h for n,h in done['files'].items())
            with np.load(out/'TRAJECTORIES.npz') as z:
                if state is not None:assert np.max(np.abs(state-z['start_ceiling_full_mw']))<1e-12
                state=z['end_ceiling_full_mw'].copy();cap_ref=z['capacity_full_mw'].copy()
            continue
        start=time.monotonic()
        with np.load(H/'days'/d/'CAPTURED_RAYS.npz') as z:
            participant=z['participant'];assert np.array_equal(part,participant)
            ids={int(t):j for j,t in enumerate(z['interval_index']) if z['control'][j]=='NO_FEEDBACK'}
            raw=np.stack([z['request_kw'][ids[t]]/1000 for t in range(48)])
            loads=np.stack([z['load_kw'][ids[t]] for t in range(48)])
        with np.load(T/'trajectories'/f'{d}.npz') as z:
            cap=np.zeros(len(part));cap[part]=z['capacity_mw']
            ref={k:z[f'beta_{beta}_{k}'].copy() for k in ['ceiling','request','award','delivery']} if is_ecf else {}
        if not is_ecf:
            with np.load(B/'days'/d/'TRAJECTORIES.npz') as z:ref={k:z[f'UNQUALIFIED_SAME_UPDATE_{k}'].copy() for k in ['ceiling','request','award','delivery']}
        if state is None:state=cap.copy()
        if cap_ref is not None:assert np.array_equal(cap,cap_ref),'Capacity changes across days'
        cap_ref=cap.copy();startstate=state.copy()
        m._WORKER['model']=m.AustralianOpenDSSModel(Path(cfg['dss_root']),connections,load_power_factor=m.LOAD_POWER_FACTOR)
        trace={k:[] for k in ['ceiling','target','request','award','delivery','available','idle','uniform_request','uniform_award','uniform_delivery','replay_award','replay_delivery']}
        records=[];ref_error=0.
        for t in range(48):
            request=np.asarray(m.apply_access_ceiling(raw[t],cap,state))
            o=m._allocate_and_realize(loads[t],gross_data[d][t],request,part)
            req,idle=m.fulfillment_replay_request(request,o['allocation_mw'],o['delivery_mw'],tolerance_mw=m.REPLAY_TOLERANCE_MW)
            assessment=None;rp=None;extra=0;qualified=False
            if is_ecf:
                rp=m._allocate_and_realize(loads[t],gross_data[d][t],np.asarray(req),part)
                assessment=m.assess_replay(allocation_mw=o['allocation_mw'],delivery_mw=o['delivery_mw'],replay_allocation_mw=rp['allocation_mw'],replay_delivery_mw=rp['delivery_mw'],idle_indices=idle,replay_valid=bool(rp['allocation'].pair_pass and rp['actual_screen'].physical_pair_pass),tolerance_mw=m.REPLAY_TOLERANCE_MW)
                qualified=assessment.gate_triggered;extra=rp['power_flow_count']
            enabled=qualified if is_ecf else bool(idle)
            after,target=m.update_access_ceiling(ceiling_before_mw=state,registered_capacity_mw=cap,delivery_mw=o['delivery_mw'],idle_indices=idle,gate_triggered=enabled,beta=beta)
            rec=m._interval_record(date=d,interval=t,control=control,raw_request_mw=raw[t],admitted_request_mw=request,outcome=o,gate=qualified,gate_reason=assessment.gate_reason if assessment else 'NOT_EVALUATED',extra_power_flows=extra,outsider_access_gain_mw=assessment.outsider_allocation_gain_mw if assessment else 0.,outsider_delivery_gain_mw=assessment.outsider_delivery_gain_mw if assessment else 0.,replay_delivery_gain_mw=assessment.total_delivery_gain_mw if assessment else 0.)
            rec.update(delivery_target_enabled=bool(enabled),beta=beta,block=job['block'],replay_evaluated=is_ecf,replay_physical_pass=bool(rp['allocation'].pair_pass and rp['actual_screen'].physical_pair_pass) if rp else None)
            records.append(rec)
            if di==0:
                for k,value in [('ceiling',state),('request',request),('award',o['allocation_mw']),('delivery',o['delivery_mw'])]:
                    e=float(np.max(np.abs(value[part]-ref[k][t])));ref_error=max(ref_error,e)
                    assert e<1e-10,(job['id'],d,t,k,e)
            mask=np.zeros(len(part),bool);mask[list(idle)]=True
            values={'ceiling':state,'target':np.asarray(target),'request':request,'award':o['allocation_mw'],'delivery':o['delivery_mw'],'available':o['available_mw'],'idle':mask}
            if is_ecf:
                ur,uf=m.matched_uniform_request(raw[t],request);ur=np.asarray(ur)
                assert abs(ur.sum()-request.sum())<1e-10
                u=m._allocate_and_realize(loads[t],gross_data[d][t],ur,part)
                uq=m._interval_record(date=d,interval=t,control='MATCHED_UNIFORM',raw_request_mw=raw[t],admitted_request_mw=ur,outcome=u,uniform_factor=uf)
                uq.update(delivery_target_enabled=False,beta=beta,block=job['block'],replay_evaluated=False,replay_physical_pass=None)
                records.append(uq)
                values.update(uniform_request=ur,uniform_award=u['allocation_mw'],uniform_delivery=u['delivery_mw'],replay_award=rp['allocation_mw'],replay_delivery=rp['delivery_mw'])
            for k,value in values.items():trace[k].append(value[part].copy())
            state=np.asarray(after)
        np.savez_compressed(out/'TRAJECTORIES.npz',capacity_full_mw=cap,participant=part,start_ceiling_full_mw=startstate,end_ceiling_full_mw=state,raw_request_mw=raw[:,part],**{k:np.stack(v) for k,v in trace.items() if v})
        result={'job':job,'date':d,'records':records,'first_block_day_reference_max_error_mw':ref_error if di==0 else None,'wall_seconds':time.monotonic()-start,'protocol_sha256':sha(PROTOCOL)}
        (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        done={'files':{n:sha(out/n) for n in ['RESULT.json','TRAJECTORIES.npz']},'protocol_sha256':sha(PROTOCOL),'all_observed_physical_pass':all(q['physical_pair_pass'] for q in records),'all_attempted_replays_physical_pass':all(q['replay_physical_pass'] is not False for q in records)}
        (out/'DONE.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
        print(json.dumps({'job':job['id'],'date':d,'day':di+1,'of':len(job['dates']),'seconds':round(result['wall_seconds'],2),'physical_pass':done['all_observed_physical_pass']},ensure_ascii=True),flush=True)
    return {'job':job['id'],'complete_days':len(job['dates'])}
def run(workers,one):
    p=verify();assert json.loads((A/'UNIT_TESTS.json').read_text())['protocol_sha256']==sha(PROTOCOL)
    jobs=p['jobs']
    if one:jobs=[j for j in jobs if j['id']==one];assert len(jobs)==1
    jobs=sorted(jobs,key=lambda j:-len(j['dates']))
    print(json.dumps({'pid':os.getpid(),'jobs':len(jobs),'workers':workers,'protocol_sha256':sha(PROTOCOL)}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(job_task,j):j['id'] for j in jobs}
        for f in as_completed(fs):
            try:print(json.dumps(f.result()),flush=True)
            except BaseException as e:
                (A/f'FAILURE_{fs[f]}.json').write_text(json.dumps({'job':fs[f],'error':repr(e),'utc':time.time()}),encoding='utf-8')
                raise
    verify();print(json.dumps({'complete_selected_jobs':len(jobs),'all_jobs':len(jobs)==len(p['jobs'])}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','test','run']);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--one',default='');a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='test':tests()
    else:run(a.workers,a.one)
