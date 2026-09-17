"""Reconstruct existing individual trajectories from fixed inputs and saved rho/g.

No allocation optimization, AC solve, new control, parameter, or efficacy case.
Original decision outputs are inputs, not independently validated predictions.
"""
from pathlib import Path
import sys,csv,json,hashlib
from collections import defaultdict
import numpy as np

A=Path(__file__).resolve().parent;F=A.parent/'151_FROZEN_SCENARIO_EVALUATION_20260907'
H=A.parent/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2'
P=A.parent.parent/'EES_PARAGRAPH_REVIEW_MASTER_20260901'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((H/'AUDIT_PROTOCOL.json').read_text('utf-8'));config=protocol['config']
sys.path.insert(0,str(F/'scientific_source/src'))
from r4r.australian_data import parse_dss_load_connections,connection_point_power
from r4r.externality_feedback import update_access_ceiling,fulfillment_replay_request,apply_access_ceiling
sources={}
for name in ['australian_data.py','externality_feedback.py']:
    path=F/'scientific_source/src/r4r'/name
    assert sha(path)==protocol['core_hashes'][name];sources[str(path)]=sha(path)
for name,digest in protocol['input_hashes'].items():
    path=Path(name)
    if path.suffix in ['.dss','.npz'] or path.name=='CSIRO_LOAD_TO_AUSGRID_PROFILE_MAPPING.csv':
        assert sha(path)==digest;sources[name]=digest
connections=parse_dss_load_connections(config['dss_root'])
with Path(config['mapping_file']).open(encoding='utf-8-sig',newline='') as f:
    mapping={r['load_name']:r for r in csv.DictReader(f)}
profile=np.array([int(mapping[x.load_name]['ausgrid_customer_id'])-1 for x in connections])
participant=np.array([mapping[x.load_name]['participant_33pct']=='1' for x in connections])
active=np.flatnonzero(participant)
assert len(connections)==1255 and len(active)==418
ev=np.load(Path(config['package'])/'AUSGRID_EVALUATION_2012_2013.npz',allow_pickle=False)
date_index={str(d):i for i,d in enumerate(ev['dates'])}
pv_all=ev['gross_generation_kw'];load_all=ev['household_load_kw']
pv_scale=5.881304516313253
capacity=np.where(participant,pv_scale*ev['generator_capacity_kwp'][profile].astype(float)/1000.,0.)
betas={'0.25':'beta_025_exact','0.5':'primary','1.0':'beta_100_exact'}
records={};dates=None;original_hashes={};max_errors=defaultdict(float)
for beta,case in betas.items():
    file=F/'runs'/case/'AU_EXTERNAL_INTERVAL_RESULTS.csv'
    manifest=json.loads((file.parent/'AU_EXTERNAL_RESULT_SHA256_MANIFEST.json').read_text('utf-8'))
    assert sha(file)==next(x['sha256'] for x in manifest['payload'] if x['path']==file.name)
    original_hashes[str(file)]=sha(file)
    with file.open(encoding='utf-8-sig',newline='') as f: rs=list(csv.DictReader(f))
    records[beta]={(r['date'],int(r['interval_index']),r['control']):r for r in rs}
    ds=sorted({r['date'] for r in rs})
    if dates is None:dates=ds
    assert dates==ds and len(rs)==80*48*4 and len(records[beta])==len(rs)

day_metrics=[];summary={};totals=defaultdict(float);bins=defaultdict(lambda:defaultdict(float))
paths=A/'trajectories';paths.mkdir(exist_ok=True)
for date in dates:
    capfile=H/'days'/date/'CAPTURED_RAYS.npz'
    classification=json.loads((capfile.parent/'REPRODUCTION_CLASSIFICATION.json').read_text('utf-8'))
    assert sha(capfile)==classification['captured_rays_sha256']
    sources[str(capfile)]=sha(capfile)
    with np.load(capfile,allow_pickle=False) as capture:
        raw_rows={(int(t),str(c)):j for j,(t,c) in enumerate(zip(capture['interval_index'],capture['control']))}
        raw=np.stack([capture['request_kw'][raw_rows[t,'NO_FEEDBACK']]/1000. for t in range(48)])
        saved_primary=np.stack([capture['request_kw'][raw_rows[t,'ECF']]/1000. for t in range(48)])
        saved_load=np.stack([capture['load_kw'][raw_rows[t,'NO_FEEDBACK']] for t in range(48)])
    idx=date_index[date]
    load=.3*load_all[idx,:,:].astype(float)[:,profile]
    pv=pv_scale*pv_all[idx,:,:].astype(float)[:,profile]
    assert np.max(np.abs(saved_load-load))<1e-10
    available=np.maximum(pv-load,0.)/1000.;available[:,~participant]=0.
    traces={}
    for beta in betas:
        ceiling=capacity.copy();trace={k:[] for k in ['ceiling','request','award','delivery','idle','tightened']}
        for t in range(48):
            r=records[beta][date,t,'ECF']
            request=np.array(apply_access_ceiling(raw[t],capacity,ceiling))
            award=float(r['request_scale'])*request
            _,_,y_kw,a_kw=connection_point_power(load[t],pv[t],award*1000.,participant,load_power_factor=.95)
            y=y_kw/1000.;a=a_kw/1000.
            assert np.max(np.abs(a-available[t]))<1e-12
            if beta=='0.5':
                error=float(np.max(np.abs(request-saved_primary[t])))
                max_errors['primary_captured_request_MW']=max(max_errors['primary_captured_request_MW'],error)
                assert error<1e-10
            quantities={'admitted_request_kw':request.sum()*1000.,'authorized_export_kw':award.sum()*1000.,'delivered_export_kw':y.sum()*1000.,'available_export_kw':a.sum()*1000.,'idle_authorization_kw':(award-y).sum()*1000.}
            for name,value in quantities.items():
                error=abs(float(value)-float(r[name]));max_errors[name]=max(max_errors[name],error)
                assert error<1e-7,(beta,date,t,name,error)
            _,idle=fulfillment_replay_request(request,award,y,tolerance_mw=2e-6)
            after,_=update_access_ceiling(ceiling_before_mw=ceiling,registered_capacity_mw=capacity,delivery_mw=y,idle_indices=idle,gate_triggered=bool(int(r['gate_triggered'])),beta=float(beta))
            after=np.array(after);mask=np.zeros(1255,dtype=bool);mask[list(idle)]=True
            for k,v in [('ceiling',ceiling),('request',request),('award',award),('delivery',y),('idle',mask),('tightened',after<ceiling-1e-12)]:
                trace[k].append(v[active].copy())
            if beta=='1.0' and not int(r['gate_triggered']):assert np.array_equal(after,capacity)
            ceiling=after
        traces[beta]={k:np.stack(v) for k,v in trace.items()}
    avail=available[:,active]
    low=traces['0.5'];high=traces['1.0']
    delta=high['delivery']-low['delivery']
    for t in range(48):
        prior_tight=np.zeros(418,dtype=bool) if t==0 else high['tightened'][t-1]
        rising=np.zeros(418,dtype=bool) if t==0 else avail[t]>avail[t-1]+1e-12
        # Excludes requested/available power before network allocation; not feasible recoverable power.
        excluded=np.minimum(raw[t,active],avail[t])>high['ceiling'][t]+1e-12
        for j in range(418):
            label=f'prior_tightening={int(prior_tight[j])};availability_rising={int(rising[j])};ceiling_excludes_power={int(excluded[j])}'
            b=bins[label];b['connection_intervals']+=1
            b['signed_delivery_difference_MWh']+=float(delta[t,j])*.5
            b['positive_delivery_difference_MWh']+=max(float(delta[t,j]),0.)*.5
            b['negative_delivery_difference_MWh']+=max(-float(delta[t,j]),0.)*.5
        totals['beta1_minus_beta05_MWh']+=float(delta[t].sum())*.5
    counts={}
    for beta,trace in traces.items():
        under=trace['ceiling']<capacity[active][None,:]-1e-12
        runs=np.zeros(418,dtype=int);maxrun=0
        for flags in under:
            runs=np.where(flags,runs+1,0);maxrun=max(maxrun,int(runs.max()))
        counts[beta]={'below_capacity_connection_intervals':int(under.sum()),'max_within_day_run_below_capacity':maxrun,'tightening_updates':int(trace['tightened'].sum())}
    day_metrics.append({'date':date,'beta1_minus_beta05_MWh':float(delta.sum())*.5,'counts':counts})
    np.savez_compressed(paths/f'{date}.npz',available_mw=avail,capacity_mw=capacity[active],raw_request_mw=raw[:,active],**{f'beta_{b}_{k}':v for b,tr in traces.items() for k,v in tr.items()})
for b in betas:
    summary[b]={'below_capacity_connection_intervals':sum(d['counts'][b]['below_capacity_connection_intervals'] for d in day_metrics),'tightening_updates':sum(d['counts'][b]['tightening_updates'] for d in day_metrics),'max_within_day_run_below_capacity':max(d['counts'][b]['max_within_day_run_below_capacity'] for d in day_metrics)}
assert abs(totals['beta1_minus_beta05_MWh']-(-4.435237553813948))<1e-8
assert abs(sum(b['signed_delivery_difference_MWh'] for b in bins.values())-totals['beta1_minus_beta05_MWh'])<1e-8
assert all(sha(Path(p))==h for p,h in original_hashes.items())
out={'scope':'Deterministic reconstruction of existing daily-reset trajectories using recorded rho and qualification. No new counterfactual efficacy or AC calculations.','days':80,'participants':418,'beta_summary':summary,'totals':dict(totals),'max_reconciliation_errors':dict(max_errors),'groups_descriptive_only':dict(bins),'original_result_hashes':original_hashes,'source_hashes':sources,'days_detail':day_metrics,'beta1_closed_condition_next_ceiling_is_capacity_verified':True}
(A/'RECONSTRUCTION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['source_hashes','days_detail']},ensure_ascii=True,indent=2))
