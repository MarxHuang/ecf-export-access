"""Frozen 80-day same-update comparison; reuse unchanged original scientific core."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent
F=R/'151_FROZEN_SCENARIO_EVALUATION_20260907'
H=R/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2'
T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
PROTOCOL=A/'PROTOCOL.json'
def original():
    src=F/'scientific_source/tools/run_ees_australian_external_replay.py'
    spec=importlib.util.spec_from_file_location('original_au_runner',src)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m
def freeze():
    assert not PROTOCOL.exists(),'Protocol exists; never overwrite a frozen version.'
    p=json.loads((H/'AUDIT_PROTOCOL.json').read_text('utf-8'))
    paths={Path(s):h for s,h in p['input_hashes'].items()}
    for n,h in p['core_hashes'].items():paths[F/'scientific_source/src/r4r'/n]=h
    paths[F/'scientific_source/tools/run_ees_australian_external_replay.py']=p['runner_sha256']
    with (F/'runs/primary/AU_EXTERNAL_INTERVAL_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:
        dates=sorted({r['date'] for r in csv.DictReader(f)})
    assert len(dates)==80
    trace=json.loads((T/'INDEPENDENT_TRAJECTORY_CHECK.json').read_text('utf-8'))['trajectory_sha256']
    for d in dates:
        file=H/'days'/d/'CAPTURED_RAYS.npz'
        paths[file]=json.loads((file.parent/'REPRODUCTION_CLASSIFICATION.json').read_text('utf-8'))['captured_rays_sha256']
        paths[T/'trajectories'/f'{d}.npz']=trace[f'{d}.npz']
    for file,h in paths.items():assert sha(file)==h,str(file)
    value={'authorization':'User explicitly authorized the new experiment after the 168 diagnostic.',
      'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dates':dates,'config':p['config'],
      'controls':['ECF','UNQUALIFIED_SAME_UPDATE'],'beta':0.5,'daily_reset':True,'interval_hours':0.5,
      'sole_control_difference':'For an unused-award member use the original delivery target regardless of replay qualification; all other targets and the update function are unchanged.',
      'data_contract':'Reuse frozen ex-ante raw requests; each control solves its own allocations, delivery and next ceiling. Do not reuse ECF future outcomes for the new control.',
      'statistics':{'primary':'paired network-day delivered-energy difference ECF minus UNQUALIFIED_SAME_UPDATE, all 80 days','secondary':['authorization','unused authorization','source-terminal exchange','curtailed availability','per-participant ceiling/delivery distributions','all physical checks','necessary power-flow counts'],'uncertainty':'Reuse original date-dependent wild bootstrap specification; do not treat intervals/connections as independent samples.'},
      'exclusions':'None based on results. Physical failures retained. No new scale, gain or date selection.',
      'computation':'New control does not run replay; ECF replay is counted in its necessary calls. Timing is descriptive, not controlled benchmarking.',
      'reference_tolerances':{'decision_power_kW':1e-7,'rho':1e-10,'recovery_kW':1e-7,'source_and_loss_kW':0.0001,'voltage_pu':1e-7,'transformer_loading_pu':1e-7},
      'inputs':{str(k):v for k,v in paths.items()},'implementation_sha256':sha(Path(__file__)),
      'protocol_notes_sha256':sha(A/'PROTOCOL.md')}
    PROTOCOL.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'dates':len(dates),'input_files':len(paths),'protocol_sha256':sha(PROTOCOL)}))
def verify_protocol():
    p=json.loads(PROTOCOL.read_text('utf-8'))
    assert sha(Path(__file__))==p['implementation_sha256']
    assert sha(A/'PROTOCOL.md')==p['protocol_notes_sha256']
    for file,h in p['inputs'].items():assert sha(Path(file))==h,file
    return p
def update(m,ceiling,capacity,delivery,idle,qualified,require_qualification):
    enabled=bool(qualified) if require_qualification else bool(idle)
    result,target=m.update_access_ceiling(ceiling_before_mw=ceiling,registered_capacity_mw=capacity,delivery_mw=delivery,idle_indices=idle,gate_triggered=enabled,beta=0.5)
    return np.asarray(result),np.asarray(target),enabled
def tests():
    verify_protocol();m=original();rng=np.random.default_rng(169);error=0.
    for k in range(1000):
        cap=rng.uniform(.001,.1,30);ceiling=rng.uniform(0,1,30)*cap
        award=rng.uniform(0,1,30)*ceiling;delivery=rng.uniform(0,1,30)*award
        if k%3==0:delivery[:]=award
        idle=tuple(np.flatnonzero(award-delivery>2e-6));qualified=bool(k%2)
        for required in [True,False]:
            actual,target,enabled=update(m,ceiling,cap,delivery,idle,qualified,required)
            expected=cap.copy()
            if (qualified if required else bool(idle)):expected[list(idle)]=delivery[list(idle)]
            assert np.allclose(target,expected,rtol=0,atol=1e-15)
            error=max(error,float(np.max(np.abs(actual-(.5*ceiling+.5*expected)))))
        gated,_,_=update(m,ceiling,cap,delivery,idle,False,True)
        ungated,_,_=update(m,ceiling,cap,delivery,idle,False,False)
        expected=np.zeros_like(cap);expected[list(idle)]=.5*(cap[list(idle)]-delivery[list(idle)])
        assert np.max(np.abs(gated-ungated-expected))<1e-15
    result={'states':1000,'updates':2000,'max_error':error,'protocol_sha256':sha(PROTOCOL)}
    (A/'UNIT_TESTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
def day_task(date):
    p=json.loads(PROTOCOL.read_text('utf-8'));m=original();cfg=p['config']
    capfile=H/'days'/date/'CAPTURED_RAYS.npz';trfile=T/'trajectories'/f'{date}.npz'
    with np.load(capfile) as z:
        participant=z['participant'];ids={int(t):j for j,t in enumerate(z['interval_index']) if z['control'][j]=='NO_FEEDBACK'}
        raw=np.stack([z['request_kw'][ids[t]]/1000 for t in range(48)])
        loads=np.stack([z['load_kw'][ids[t]] for t in range(48)])
    with np.load(trfile) as z:
        capacity=np.zeros(len(participant));capacity[participant]=z['capacity_mw']
        reftrace={k:z[f'beta_0.5_{k}'].copy() for k in ['ceiling','request','award','delivery']}
    connections=m.parse_dss_load_connections(cfg['dss_root'])
    profile,part=m._load_mapping(Path(cfg['mapping_file']),connections);assert np.array_equal(part,participant)
    with np.load(Path(cfg['package'])/'AUSGRID_EVALUATION_2012_2013.npz') as z:
        i=list(map(str,z['dates'])).index(date)
        gross=m.PV_SCALE*z['gross_generation_kw'][i,:,:].astype(float)[:,profile]
        expected_load=.3*z['household_load_kw'][i,:,:].astype(float)[:,profile]
    assert np.max(np.abs(loads-expected_load))<1e-10
    with (F/'runs/primary/AU_EXTERNAL_INTERVAL_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:
        old={int(r['interval_index']):r for r in csv.DictReader(f) if r['date']==date and r['control']=='ECF'}
    records=[];traces={};reference_errors={};started=time.monotonic()
    for control in p['controls']:
        m._WORKER['model']=m.AustralianOpenDSSModel(Path(cfg['dss_root']),connections,load_power_factor=m.LOAD_POWER_FACTOR)
        ceiling=capacity.copy();trace={k:[] for k in ['ceiling','target','request','award','delivery','idle']}
        for t in range(48):
            request=np.asarray(m.apply_access_ceiling(raw[t],capacity,ceiling))
            outcome=m._allocate_and_realize(loads[t],gross[t],request,participant)
            replay_req,idle=m.fulfillment_replay_request(request,outcome['allocation_mw'],outcome['delivery_mw'],tolerance_mw=m.REPLAY_TOLERANCE_MW)
            qualified=False;extra=0;assessment=None
            if control=='ECF':
                replay=m._allocate_and_realize(loads[t],gross[t],np.asarray(replay_req),participant)
                valid=bool(replay['allocation'].pair_pass and replay['actual_screen'].physical_pair_pass)
                assessment=m.assess_replay(allocation_mw=outcome['allocation_mw'],delivery_mw=outcome['delivery_mw'],replay_allocation_mw=replay['allocation_mw'],replay_delivery_mw=replay['delivery_mw'],idle_indices=idle,replay_valid=valid,tolerance_mw=m.REPLAY_TOLERANCE_MW)
                qualified=assessment.gate_triggered;extra=replay['power_flow_count']
            after,target,enabled=update(m,ceiling,capacity,outcome['delivery_mw'],idle,qualified,control=='ECF')
            rec=m._interval_record(date=date,interval=t,control=control,raw_request_mw=raw[t],admitted_request_mw=request,outcome=outcome,gate=qualified,gate_reason=assessment.gate_reason if assessment else 'NOT_EVALUATED',extra_power_flows=extra,outsider_access_gain_mw=assessment.outsider_allocation_gain_mw if assessment else 0.,outsider_delivery_gain_mw=assessment.outsider_delivery_gain_mw if assessment else 0.,replay_delivery_gain_mw=assessment.total_delivery_gain_mw if assessment else 0.)
            rec['replay_qualified']=rec.pop('gate_triggered') if control=='ECF' else None
            rec.pop('gate_triggered',None)
            rec['delivery_target_enabled']=enabled;rec['unused_award_count']=len(idle)
            rec['tightened_members']=int(np.sum(after<ceiling-1e-12));rec['replay_power_flow_count']=extra
            if control=='ECF':
                assert rec['replay_qualified']==int(old[t]['gate_triggered'])
                for key in ['gate_reason','allocation_status','physical_pair_pass','allocation_pair_pass']:
                    assert str(rec[key])==old[t][key],(date,t,key,rec[key],old[t][key])
                for key in ['raw_request_kw','admitted_request_kw','authorized_export_kw','delivered_export_kw','idle_authorization_kw','outsider_access_gain_kw','outsider_delivery_gain_kw','replay_delivery_gain_kw','request_scale','actual_source_terminal_p_kw','actual_circuit_loss_kw','actual_voltage_min_pu','actual_voltage_max_pu','actual_transformer_max_loading_pu']:
                    tol=1e-7
                    if key=='request_scale':tol=1e-10
                    if key in ['actual_source_terminal_p_kw','actual_circuit_loss_kw']:tol=.0001
                    e=abs(float(rec[key])-float(old[t][key]));reference_errors[key]=max(reference_errors.get(key,0.),e)
                    assert e<=tol,(date,t,key,e,tol)
                for key,value in [('ceiling',ceiling),('request',request),('award',outcome['allocation_mw']),('delivery',outcome['delivery_mw'])]:
                    assert np.max(np.abs(value[participant]-reftrace[key][t]))<1e-10,(date,t,key)
            mask=np.zeros(len(participant),dtype=bool);mask[list(idle)]=True
            for key,value in [('ceiling',ceiling),('target',target),('request',request),('award',outcome['allocation_mw']),('delivery',outcome['delivery_mw']),('idle',mask)]:trace[key].append(value[participant].copy())
            records.append(rec);ceiling=after
        traces.update({control+'_'+k:np.stack(v) for k,v in trace.items()})
    folder=A/'days'/date;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'DONE.json').exists()
    np.savez_compressed(folder/'TRAJECTORIES.npz',capacity_mw=capacity[participant],raw_request_mw=raw[:,participant],**traces)
    result={'date':date,'records':records,'reference_errors':reference_errors,'wall_seconds':time.monotonic()-started,'protocol_sha256':sha(PROTOCOL)}
    (folder/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    done={'date':date,'files':{n:sha(folder/n) for n in ['TRAJECTORIES.npz','RESULT.json']},'protocol_sha256':sha(PROTOCOL),'all_physical_pass':all(r['physical_pair_pass'] for r in records)}
    (folder/'DONE.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
    return {'date':date,'seconds':result['wall_seconds'],'all_physical_pass':done['all_physical_pass']}
def run(limit,workers):
    p=verify_protocol();assert json.loads((A/'UNIT_TESTS.json').read_text())['protocol_sha256']==sha(PROTOCOL)
    dates=p['dates'][:limit] if limit else p['dates'];pending=[]
    for date in dates:
        file=A/'days'/date/'DONE.json'
        if file.exists():
            d=json.loads(file.read_text());assert d['protocol_sha256']==sha(PROTOCOL)
            assert all(sha(file.parent/n)==h for n,h in d['files'].items())
        else:pending.append(date)
    print(json.dumps({'pid':os.getpid(),'selected':len(dates),'pending':len(pending),'workers':workers}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(day_task,d):d for d in pending}
        for f in as_completed(futures):
            try:print(json.dumps(f.result()),flush=True)
            except BaseException as e:
                (A/'FAILURE.json').write_text(json.dumps({'date':futures[f],'error':repr(e)}),encoding='utf-8')
                for future in futures:future.cancel()
                raise
    verify_protocol()
    print(json.dumps({'complete_selected_days':len(dates),'full_protocol_complete':len(dates)==80}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','test','run']);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='test':tests()
    else:run(args.limit,args.workers)
