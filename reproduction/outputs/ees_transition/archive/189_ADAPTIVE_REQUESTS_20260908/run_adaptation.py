"""Separate local learning histories under otherwise frozen physical controls."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from participant_policy import LocalResponse,random_draws,ACTIONS,EXPLORATION

A=Path(__file__).resolve().parent;R=A.parent
F=R/'151_FROZEN_SCENARIO_EVALUATION_20260907'
H=R/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2'
T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907'
B=R/'169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908'
C=R/'173_CONTIGUOUS_STATE_FEEDBACK_20260908'
BRANCHES=['NO_FEEDBACK','ECF','UNQUALIFIED_SAME_UPDATE']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def load_runner():
    s=importlib.util.spec_from_file_location('original_au_runner',F/'scientific_source/tools/run_ees_australian_external_replay.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def freeze():
    assert not (A/'PROTOCOL.json').exists(),'Never overwrite a frozen protocol'
    old=read(C/'PROTOCOL.json');tests=read(A/'POLICY_TESTS.json');assert tests['passed']
    inputs=dict(old['inputs'])
    for path in [C/'PROTOCOL.json',F/'runs/primary/AU_EXTERNAL_INTERVAL_RESULTS.csv']:
        inputs[str(path)]=sha(path)
    for path,h in inputs.items():assert sha(path)==h,path
    jobs=[]
    for mode in ['ADAPTIVE','STATIC_RANDOM']:
        for seed in [2001,2002,2003]:
            for i,days in enumerate(old['blocks']):
                jobs.append({'id':f'{mode}_{seed}_block{i+1:02}','mode':mode,'seed':seed,'block':i+1,'dates':days})
    p={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'config':old['config'],
       'dates':old['dates'],'blocks':old['blocks'],'jobs':jobs,'actions':ACTIONS.tolist(),'exploration':EXPLORATION,
       'uncertainty':{'replicates':10000,'seed':189,'calendar_kernel_days':8,'antithetic':True,'average_seeds_before_date_uncertainty':True},
       'inputs':inputs,'implementation':{n:sha(A/n) for n in ['run_adaptation.py','participant_policy.py','test_policy.py','PROTOCOL.md','POLICY_TESTS.json']}}
    write(A/'PROTOCOL.json',p)
    print(json.dumps({'protocol_sha256':sha(A/'PROTOCOL.json'),'jobs':len(jobs),'setting_days':sum(len(j['dates']) for j in jobs),'inputs':len(inputs)}),flush=True)

def verify():
    p=read(A/'PROTOCOL.json')
    for n,h in p['implementation'].items():assert sha(A/n)==h,n
    for n,h in p['inputs'].items():assert sha(n)==h,n
    return p

def execute(job,smoke=False):
    p=read(A/'PROTOCOL.json');ph=sha(A/'PROTOCOL.json');cfg=p['config'];m=load_runner()
    connections=m.parse_dss_load_connections(cfg['dss_root'])
    profile,part=m._load_mapping(Path(cfg['mapping_file']),connections)
    with np.load(Path(cfg['package'])/'AUSGRID_EVALUATION_2012_2013.npz') as z:
        indices={str(d):i for i,d in enumerate(z['dates'])}
        gross={d:m.PV_SCALE*z['gross_generation_kw'][indices[d]].astype(float)[:,profile] for d in job['dates']}
    with np.load(T/'trajectories'/f"{job['dates'][0]}.npz") as z:
        cap=np.zeros(len(part));cap[part]=z['capacity_mw']
    assert part.sum()==418 and len(part)==1255
    policies={c:LocalResponse(cap[part],job['mode']) for c in BRANCHES}
    states={c:cap.copy() for c in BRANCHES}
    root=A/('smoke' if smoke else 'jobs')/job['id'];root.mkdir(parents=True,exist_ok=True)
    original={}
    if smoke:
        with (F/'runs/primary/AU_EXTERNAL_INTERVAL_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:
            original={(q['date'],int(q['interval_index']),q['control']):q for q in csv.DictReader(f)}
    for day_index,d in enumerate(job['dates']):
        out=root/d;out.mkdir(exist_ok=True)
        if (out/'DONE.json').exists():
            done=read(out/'DONE.json');assert done['protocol_sha256']==ph
            for n,h in done['files'].items():assert sha(out/n)==h
            with np.load(out/'TRAJECTORIES.npz') as z:
                assert np.array_equal(z['capacity_mw'],cap[part])
                for c in BRANCHES:
                    assert np.array_equal(states[c][part],z[c+'_start_ceiling'])
                    assert np.array_equal(policies[c].state(),z[c+'_start_weights'])
                    states[c][part]=z[c+'_end_ceiling']
                    policies[c].restore(z[c+'_end_weights'],48*(day_index+1))
            continue
        start=time.monotonic()
        with np.load(H/'days'/d/'CAPTURED_RAYS.npz') as z:
            assert np.array_equal(part,z['participant'])
            ids={int(t):i for i,t in enumerate(z['interval_index']) if z['control'][i]=='NO_FEEDBACK'}
            assert sorted(ids)==list(range(48))
            base=np.stack([z['request_kw'][ids[t]]/1000 for t in range(48)])
            loads=np.stack([z['load_kw'][ids[t]] for t in range(48)])
        refs={}
        if smoke:
            with np.load(T/'trajectories'/f'{d}.npz') as z:
                refs['ECF']={k:z[f'beta_0.5_{k}'].copy() for k in ['ceiling','request','award','delivery']}
            with np.load(B/'days'/d/'TRAJECTORIES.npz') as z:
                refs['UNQUALIFIED_SAME_UPDATE']={k:z[f'UNQUALIFIED_SAME_UPDATE_{k}'].copy() for k in ['ceiling','request','award','delivery']}
        initial={}
        for c in BRANCHES:
            initial[c+'_start_ceiling']=states[c][part].copy();initial[c+'_start_weights']=policies[c].state()
        m._WORKER['model']=m.AustralianOpenDSSModel(Path(cfg['dss_root']),connections,load_power_factor=m.LOAD_POWER_FACTOR)
        traces={};records=[];ref_error=0.;record_error=0.
        def trace(c,k,x):traces.setdefault(c+'_'+k,[]).append(np.asarray(x).copy())
        for t in range(48):
            u=random_draws(job['seed'],d,t,int(part.sum()))
            # Choices are made from forecast requests and local past rewards only.
            # Contemporary physical truth is passed only to the network solver.
            choices={c:policies[c].choose(base[t,part],u) for c in BRANCHES}
            for c in BRANCHES:
                chosen,action,probs=choices[c];raw=np.zeros(len(part));raw[part]=chosen
                request=raw.copy() if c=='NO_FEEDBACK' else np.asarray(m.apply_access_ceiling(raw,cap,states[c]))
                o=m._allocate_and_realize(loads[t],gross[d][t],request,part)
                req,idle=m.fulfillment_replay_request(request,o['allocation_mw'],o['delivery_mw'],tolerance_mw=m.REPLAY_TOLERANCE_MW)
                rp=None;assessment=None;enabled=False
                if c=='ECF':
                    rp=m._allocate_and_realize(loads[t],gross[d][t],np.asarray(req),part)
                    assessment=m.assess_replay(allocation_mw=o['allocation_mw'],delivery_mw=o['delivery_mw'],replay_allocation_mw=rp['allocation_mw'],replay_delivery_mw=rp['delivery_mw'],idle_indices=idle,replay_valid=bool(rp['allocation'].pair_pass and rp['actual_screen'].physical_pair_pass),tolerance_mw=m.REPLAY_TOLERANCE_MW)
                    enabled=assessment.gate_triggered
                elif c=='UNQUALIFIED_SAME_UPDATE':enabled=bool(idle)
                if c=='NO_FEEDBACK':after=cap.copy();target=cap.copy()
                else:
                    after,target=m.update_access_ceiling(ceiling_before_mw=states[c],registered_capacity_mw=cap,delivery_mw=o['delivery_mw'],idle_indices=idle,gate_triggered=enabled,beta=cfg['beta'])
                    after=np.asarray(after);target=np.asarray(target)
                rec=m._interval_record(date=d,interval=t,control=c,raw_request_mw=raw,admitted_request_mw=request,outcome=o,gate=bool(assessment and assessment.gate_triggered),gate_reason=assessment.gate_reason if assessment else 'NOT_EVALUATED',extra_power_flows=rp['power_flow_count'] if rp else 0,outsider_access_gain_mw=assessment.outsider_allocation_gain_mw if assessment else 0.,outsider_delivery_gain_mw=assessment.outsider_delivery_gain_mw if assessment else 0.,replay_delivery_gain_mw=assessment.total_delivery_gain_mw if assessment else 0.)
                rec.update(mode=job['mode'],seed=job['seed'],block=job['block'],delivery_target_enabled=bool(enabled),replay_physical_pass=bool(rp['allocation'].pair_pass and rp['actual_screen'].physical_pair_pass) if rp else None)
                records.append(rec)
                mask=np.zeros(len(part),bool);mask[list(idle)]=True
                for k,v in {'raw':raw,'ceiling':states[c],'target':target,'request':request,'award':o['allocation_mw'],'delivery':o['delivery_mw'],'available':o['available_mw'],'idle':mask}.items():trace(c,k,v[part])
                trace(c,'action',action);trace(c,'probabilities',probs)
                reward=policies[c].observe(o['delivery_mw'][part]);trace(c,'reward',reward)
                if smoke:
                    for k,v in [('ceiling',states[c]),('request',request),('award',o['allocation_mw']),('delivery',o['delivery_mw'])]:
                        if c in refs:
                            e=float(np.max(np.abs(v[part]-refs[c][k][t])));ref_error=max(e,ref_error)
                            assert e<1e-10,(d,t,c,k,e)
                    if c in ['NO_FEEDBACK','ECF']:
                        for k in ['raw_request_kw','admitted_request_kw','authorized_export_kw','delivered_export_kw','idle_authorization_kw']:
                            e=abs(float(rec[k])-float(original[(d,t,c)][k]));record_error=max(record_error,e)
                            assert e<1e-7,(d,t,c,k,e)
                states[c]=after
                if c=='ECF':
                    ur,uf=m.matched_uniform_request(raw,request);ur=np.asarray(ur)
                    assert abs(ur.sum()-request.sum())<1e-10
                    uniform=m._allocate_and_realize(loads[t],gross[d][t],ur,part)
                    uq=m._interval_record(date=d,interval=t,control='MATCHED_UNIFORM_SHADOW',raw_request_mw=raw,admitted_request_mw=ur,outcome=uniform,uniform_factor=uf)
                    uq.update(mode=job['mode'],seed=job['seed'],block=job['block'],delivery_target_enabled=False,replay_physical_pass=None)
                    records.append(uq)
                    for k,v in {'request':ur,'award':uniform['allocation_mw'],'delivery':uniform['delivery_mw']}.items():trace('MATCHED_UNIFORM_SHADOW',k,v[part])
                    for k,v in {'request':np.asarray(req),'award':rp['allocation_mw'],'delivery':rp['delivery_mw']}.items():trace('REPLAY',k,v[part])
        for c in BRANCHES:
            initial[c+'_end_ceiling']=states[c][part].copy();initial[c+'_end_weights']=policies[c].state()
        np.savez_compressed(out/'TRAJECTORIES.npz',participant=part,capacity_mw=cap[part],base_request_mw=base[:,part],**initial,**{k:np.stack(v) for k,v in traces.items()})
        result={'job':job,'date':d,'records':records,'wall_seconds':time.monotonic()-start,'reference_max_error_mw':ref_error if smoke else None,'reference_record_error_kw':record_error if smoke else None,'protocol_sha256':ph}
        write(out/'RESULT.json',result)
        done={'protocol_sha256':ph,'files':{n:sha(out/n) for n in ['TRAJECTORIES.npz','RESULT.json']},'all_observed_physical_pass':all(q['physical_pair_pass'] for q in records),'all_replays_physical_pass':all(q['replay_physical_pass'] is not False for q in records)}
        write(out/'DONE.json',done)
        print(json.dumps({'job':job['id'],'date':d,'seconds':round(result['wall_seconds'],2),'physical_pass':done['all_observed_physical_pass'],'replay_pass':done['all_replays_physical_pass']},ensure_ascii=True),flush=True)
    return {'job':job['id'],'completed_days':len(job['dates'])}

def smoke():
    p=verify();job={'id':'FIXED_REFERENCE','mode':'FIXED_REFERENCE','seed':2001,'block':1,'dates':[p['dates'][0]]}
    execute(job,True)
    result=read(A/'smoke/FIXED_REFERENCE'/p['dates'][0]/'RESULT.json')
    verify();write(A/'SMOKE_TEST.json',{'passed':True,'protocol_sha256':sha(A/'PROTOCOL.json'),'reference_max_error_mw':result['reference_max_error_mw'],'reference_record_error_kw':result['reference_record_error_kw'],'date':p['dates'][0]})

def run(workers):
    p=verify();assert 1<=workers<=4
    s=read(A/'SMOKE_TEST.json');assert s['passed'] and s['protocol_sha256']==sha(A/'PROTOCOL.json')
    jobs=sorted(p['jobs'],key=lambda j:-len(j['dates']))
    print(json.dumps({'pid':os.getpid(),'jobs':len(jobs),'workers':workers,'protocol_sha256':sha(A/'PROTOCOL.json')}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(execute,j):j['id'] for j in jobs}
        for f in as_completed(fs):
            try:print(json.dumps(f.result()),flush=True)
            except BaseException as e:
                write(A/f'FAILURE_{fs[f]}.json',{'job':fs[f],'error':repr(e),'utc':time.time()});raise
    verify();write(A/'RUN_COMPLETE.json',{'protocol_sha256':sha(A/'PROTOCOL.json'),'completed_jobs':len(jobs),'setting_days':480,'independent_observed_dates':80})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','smoke','run','verify']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='smoke':smoke()
    elif args.action=='verify':print({'inputs_and_code_verified':bool(verify())})
    else:run(args.workers)
