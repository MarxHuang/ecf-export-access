"""Independent point-record and lower-bound feedback trajectories."""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from control_step import choose_transition,load_module,core
A=Path(__file__).resolve().parent;R=A.parent
F=R/'151_FROZEN_SCENARIO_EVALUATION_20260907'
H=R/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2'
T=R/'164_CEILING_TRAJECTORY_REVIEW_20260907'
P=R.parent/'EES_PARAGRAPH_REVIEW_MASTER_20260901'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def runner():
    return load_module('frozen_au_151_bounded',F/'scientific_source/tools/run_ees_australian_external_replay.py')
def freeze():
    assert not (A/'PROTOCOL.json').exists(),'Do not overwrite a frozen experiment'
    old=json.loads((R/'173_CONTIGUOUS_STATE_FEEDBACK_20260908/PROTOCOL.json').read_text('utf-8'))
    inputs=dict(old['inputs'])
    for n in ['v2/bounded_feedback.py','v2/PROTOCOL.json','v2/FIXED_PAIR_RESULTS.json','INDEPENDENT_CHECK.json']:
        file=R/'175_BOUNDED_RECORD_INTERFACE_20260908'/n;inputs[str(file)]=sha(file)
    for file,h in inputs.items():assert sha(Path(file))==h,file
    cases=[{'id':'EXACT','fraction':0.,'sign':0,'mode':'LOWER_BOUND'}]
    for f in [.01,.05,.10]:
        for sign in [-1,1]:
            for mode in ['POINT','LOWER_BOUND']:
                cases.append({'id':f"R{int(f*100):02}_{'POS' if sign>0 else 'NEG'}_{mode}",'fraction':f,'sign':sign,'mode':mode})
    p={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dates':old['dates'],
       'cases':cases,'config':old['config'],'gain':.5,'daily_capacity_initialization':True,
       'inputs':inputs,'implementation':{n:sha(A/n) for n in ['PROTOCOL.md','control_step.py','run_feedback.py','test_information_path.py']},
       'paper_at_start':{n:sha(P/n) for n in ['main.tex','supporting_information.tex','main.pdf','supporting_information.pdf','EES_OVERLEAF_CURRENT_20260907.zip']},
       'expected':{'setting_days':1040,'observed_records':49920,'replay_records':49920},
       'analysis':{'seed':176,'replicates':10000,'kernel':'max(1 - abs(calendar gap)/8,0)','retain_all_cases':True},
       'scope':'Synthetic bounded record error, independent feedback trajectories, record-based replay AC and separate true-state AC audit; no instrument calibration.'}
    (A/'PROTOCOL.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(cases),'days':len(p['dates']),'source_files':len(inputs),'hash':sha(A/'PROTOCOL.json')}))
def verify():
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'))
    for n,h in p['implementation'].items():assert sha(A/n)==h,n
    for n,h in p['inputs'].items():assert sha(Path(n))==h,n
    return p
def task(case,smoke=False):
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'));cfg=p['config'];m=runner()
    connections=m.parse_dss_load_connections(cfg['dss_root'])
    profile,part=m._load_mapping(Path(cfg['mapping_file']),connections)
    dates=p['dates'][:1] if smoke else p['dates']
    with np.load(Path(cfg['package'])/'AUSGRID_EVALUATION_2012_2013.npz') as z:
        ids={str(d):i for i,d in enumerate(z['dates'])}
        gross_by_day={d:m.PV_SCALE*z['gross_generation_kw'][ids[d]].astype(float)[:,profile] for d in dates}
    completed=0
    for date in dates:
        out=A/'cases'/case['id']/date;out.mkdir(parents=True,exist_ok=True)
        if (out/'DONE.json').exists():
            done=json.loads((out/'DONE.json').read_text('utf-8'))
            assert done['protocol_sha256']==sha(A/'PROTOCOL.json')
            assert all(sha(out/n)==h for n,h in done['files'].items())
            completed+=1;continue
        begin=time.monotonic()
        with np.load(H/'days'/date/'CAPTURED_RAYS.npz') as z:
            assert np.array_equal(part,z['participant'])
            ix={int(t):k for k,t in enumerate(z['interval_index']) if z['control'][k]=='NO_FEEDBACK'}
            raw=np.stack([z['request_kw'][ix[t]]/1000 for t in range(48)])
            loads=np.stack([z['load_kw'][ix[t]] for t in range(48)])
        with np.load(T/'trajectories'/f'{date}.npz') as z:
            cap=np.zeros(part.size);cap[part]=z['capacity_mw']
            ref={k:z[f'beta_0.5_{k}'].copy() for k in ['ceiling','request','award','delivery']}
            ref_idle=z['beta_0.5_idle'].copy()
        delta=case['fraction']*cap;state=cap.copy()
        model=m.AustralianOpenDSSModel(Path(cfg['dss_root']),connections,load_power_factor=m.LOAD_POWER_FACTOR)
        m._WORKER['model']=model
        records=[];traces={k:[] for k in ['ceiling','target','request','award','delivery','available',
           'record','error_radius','replay_award','record_replay_delivery','lower_replay_delivery','truth_replay_delivery']}
        imports=[];max_exact_error=0.
        for t in range(48):
            request=np.asarray(m.apply_access_ceiling(raw[t],cap,state))
            # External environment produces completed metering and test records.
            o=m._allocate_and_realize(loads[t],gross_by_day[date][t],request,part)
            actual_a=o['available_mw'];y=o['delivery_mw'];x=o['allocation_mw']
            metered_import=np.where(part,np.maximum(loads[t]-gross_by_day[date][t],0.),loads[t])
            h=np.maximum(y,actual_a+case['sign']*delta)
            assert np.max(np.abs(h-actual_a)-delta)<1e-12
            step=choose_transition(runner=m,model=model,load_kw=loads[t],net_import_kw=metered_import,
                participant=part,request=request,allocation=x,delivery=y,capacity=cap,ceiling=state,
                record=h,error_radius=delta,mode=case['mode'])
            q=step['assessment'];after=step['after']
            # Outcome-only diagnostics: nothing below is supplied to choose_transition.
            diag=dict(allocation=x,delivery=y,replay_allocation=step['replay_allocation'],record_valid=True,
                replay_physical_valid=True,tolerance=m.REPLAY_TOLERANCE_MW)
            truth=core.assess_bounded_record(**diag,record=actual_a,error_radius=np.zeros_like(cap))
            point=core.assess_bounded_record(**diag,record=h,error_radius=np.zeros_like(cap))
            lower=core.assess_bounded_record(**diag,record=h,error_radius=delta)
            assert not lower.qualifies or truth.qualifies
            if case['mode']=='LOWER_BOUND':assert not q.qualifies or truth.qualifies
            if case['id']=='EXACT':
                for k,arr in [('ceiling',state),('request',request),('award',x),('delivery',y)]:
                    error=float(np.max(np.abs(arr[part]-ref[k][t])));max_exact_error=max(max_exact_error,error)
                    assert error<1e-10,(case['id'],date,t,k,error)
                assert np.array_equal(x[part]-y[part]>m.REPLAY_TOLERANCE_MW,ref_idle[t])
            row=m._interval_record(date=date,interval=t,control=case['id'],raw_request_mw=raw[t],
                admitted_request_mw=request,outcome=o,gate=q.qualifies,gate_reason=q.reason,
                outsider_access_gain_mw=q.outside_access_gain,outsider_delivery_gain_mw=q.outside_delivery_gain,
                replay_delivery_gain_mw=q.total_delivery_gain,extra_power_flows=step['extra_power_flows'],
                post_event_availability_signal_mw=h)
            row.update(record_replay_physical_pass=step['replay_physical_pass'],
                record_replay_screen=asdict(step['replay_record_screen']),
                replay_authorization_pass=bool(step['replay_allocation_result'].pair_pass),
                truth_delivery_qualifies=truth.qualifies,point_delivery_qualifies=point.qualifies,
                lower_delivery_qualifies=lower.qualifies,
                false_recovery_tightening=bool(q.qualifies and not truth.qualifies),
                truth_replay_gain_MW=truth.total_delivery_gain)
            records.append(row)
            vectors={'ceiling':state,'target':step['target'],'request':request,'award':x,'delivery':y,
                'available':actual_a,'record':h,'error_radius':delta,'replay_award':step['replay_allocation'],
                'record_replay_delivery':step['record_replay_delivery'],
                'lower_replay_delivery':np.asarray(lower.lower_replay_delivery),
                'truth_replay_delivery':np.minimum(step['replay_allocation'],actual_a)}
            for k,arr in vectors.items():traces[k].append(arr[part].copy())
            imports.append(metered_import.copy());state=after
        np.savez_compressed(out/'TRAJECTORIES.npz',capacity_full_MW=cap,participant=part,
            end_ceiling_full_MW=state,raw_request_mw=raw[:,part],metered_import_kw=np.stack(imports),
            **{k:np.stack(v) for k,v in traces.items()})
        result={'case':case,'date':date,'records':records,'max_exact_reference_error_MW':max_exact_error if case['id']=='EXACT' else None,
            'wall_seconds':time.monotonic()-begin,'protocol_sha256':sha(A/'PROTOCOL.json')}
        (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        done={'protocol_sha256':sha(A/'PROTOCOL.json'),'files':{n:sha(out/n) for n in ['RESULT.json','TRAJECTORIES.npz']},
              'observed_physical_failures':sum(not r['physical_pair_pass'] for r in records),
              'record_replay_physical_failures':sum(not r['record_replay_physical_pass'] for r in records)}
        (out/'DONE.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
        completed+=1
        print(json.dumps({'case':case['id'],'date':date,'day':completed,'of':len(dates),
            'seconds':round(result['wall_seconds'],1),'physical_failures':done['observed_physical_failures']}),flush=True)
    return {'case':case['id'],'completed_days':completed}
def run(workers,smoke,one):
    p=verify();cases=p['cases']
    tests=json.loads((A/'INFORMATION_PATH_TESTS.json').read_text('utf-8'))
    assert tests['control_step_sha256']==sha(A/'control_step.py')
    if smoke:cases=[c for c in cases if c['id'] in ['EXACT','R10_POS_POINT','R10_NEG_LOWER_BOUND']]
    if one:cases=[c for c in cases if c['id']==one];assert len(cases)==1
    print(json.dumps({'pid':os.getpid(),'cases':len(cases),'workers':workers,'smoke':smoke,'protocol_sha256':sha(A/'PROTOCOL.json')}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(task,c,smoke):c['id'] for c in cases}
        for f in as_completed(futures):
            try:print(json.dumps(f.result()),flush=True)
            except BaseException as e:
                (A/f"FAILURE_{futures[f]}.json").write_text(json.dumps({'case':futures[f],'error':repr(e),'utc':time.time()}),encoding='utf-8')
                raise
    verify();print(json.dumps({'complete':True,'smoke':smoke,'cases':len(cases)}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--one',default='');args=ap.parse_args()
    freeze() if args.action=='freeze' else run(args.workers,args.smoke,args.one)
