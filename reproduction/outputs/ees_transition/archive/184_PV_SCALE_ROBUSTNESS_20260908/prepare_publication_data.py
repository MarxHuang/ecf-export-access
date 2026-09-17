"""Combine verified scale outcomes and the unchanged primary reference.

Produces JSON for later table/Origin exporters; incomplete runs are rejected.
Never writes manuscript, figure, workbook, protocol or experiment files.
"""
from pathlib import Path
import argparse,json
import numpy as np
import run_scale as run
from scale_inputs import near
A=Path(__file__).resolve().parent
CONTROLS=['NO_FEEDBACK','PERSISTENCE_ONLY','ECF','MATCHED_UNIFORM']
FIELDS=['delivered_export_mwh','authorized_export_mwh','idle_authorization_mwh','source_terminal_MWh','gate_count','power_flow_count']

def prerequisites():
    missing=[n for n in ['PIPELINE_COMPLETE.json','VERIFIED_SCALE_RESULTS.json','PRIMARY_NORMALIZATION.json','SAVED_CHECKER_TESTS.json'] if not (A/n).exists()]
    if missing:return dict(ready=False,missing=missing,publication_data_written=False)
    p=run.verify_sources()
    finish=json.loads((A/'PIPELINE_COMPLETE.json').read_text('utf-8'))
    v=json.loads((A/'VERIFIED_SCALE_RESULTS.json').read_text('utf-8'))
    assert finish['protocol_sha256']==run.sha(A/'PROTOCOL.json')
    assert finish['verified_results_sha256']==run.sha(A/'VERIFIED_SCALE_RESULTS.json')
    assert v['complete'] and v['verified_days']==320 and v['control_intervals']==61440
    assert v['post_decision_AC_replays']==v['replay_intervals']==15360 and not v['contradictions']
    assert len(v['daily'])==1280
    # Physical failures, if present, are outcomes and must not be silently excluded.
    assert len(v['failures'])==finish['physical_failure_records']
    test=json.loads((A/'SAVED_CHECKER_TESTS.json').read_text('utf-8'))
    assert test['passed'] and test['protocol_sha256']==run.sha(A/'PROTOCOL.json')
    assert test['test_sha256']==run.sha(A/'test_saved_checks.py')
    return dict(ready=True,publication_data_written=False)

def main():
    ready=prerequisites()
    assert ready['ready'],ready
    p=run.verify_sources();v=json.loads((A/'VERIFIED_SCALE_RESULTS.json').read_text('utf-8'))
    primary=json.loads((A/'PRIMARY_NORMALIZATION.json').read_text('utf-8'))
    assert primary['protocol_sha256']==run.sha(A/'PROTOCOL.json')
    for name,h in primary['source_hashes'].items():assert run.sha(Path(name))==h,name
    source_hashes={str(A/n):run.sha(A/n) for n in ['PROTOCOL.json','VERIFIED_SCALE_RESULTS.json','PIPELINE_COMPLETE.json','PRIMARY_NORMALIZATION.json','SAVED_CHECKER_TESTS.json']}
    source_hashes.update(primary['source_hashes'])
    settings=[dict(id='PV100',factor=1.,is_primary=True),*[dict(**r,is_primary=False) for r in p['settings']]]
    settings.sort(key=lambda r:r['factor'])
    daily=list(v['daily']);failure_set={(r['setting'],r['date'],r['interval'],r['control']) for r in v['failures']}
    for path in primary['source_hashes']:
        if Path(path).parent.name!='day_checkpoints':continue
        r=json.loads(Path(path).read_text('utf-8'))['result']
        assert r['date'] in p['dates']
        for s in r['day_summaries']:
            records=[q for q in r['interval_records'] if q['control']==s['control']]
            assert len(records)==48
            daily.append(dict(setting='PV100',**s,source_terminal_MWh=sum(q['actual_source_terminal_p_kw'] for q in records)*.5/1000))
            for q in records:
                if not q['physical_pair_pass']:failure_set.add(('PV100',q['date'],q['interval_index'],q['control']))
    index={(r['setting'],r['date'],r['control']):r for r in daily}
    expected={(s['id'],d,c) for s in settings for d in p['dates'] for c in CONTROLS}
    assert len(daily)==len(index)==1600 and set(index)==expected
    days=np.array(p['dates'],dtype='datetime64[D]').astype(int)
    kernel=np.maximum(1-np.abs(days[:,None]-days[None,:])/8,0)
    eigen,rotation=np.linalg.eigh(kernel)
    weights=np.random.default_rng(184).standard_normal((5000,80))@(rotation*np.sqrt(np.maximum(eigen,0))).T
    weights=np.concatenate([weights,-weights])
    energy=[];contrasts=[];exposures=[];totals={}
    for s in settings:
        name=s['id'];totals[name]={}
        capacity=primary['participant_capacity_MW']*s['factor']
        exp=v['exposure'][name] if name!='PV100' else dict(participant_capacity_MW=primary['participant_capacity_MW'],
            available_export_MWh=primary['available_MWh'],
            no_feedback_request_limited_intervals=primary['no_feedback_request_limited_intervals'],
            no_feedback_undelivered_available_MWh=primary['no_feedback_undelivered_available_MWh'])
        near(capacity,exp['participant_capacity_MW'])
        for c in CONTROLS:
            selected=[index[name,d,c] for d in p['dates']]
            for r in selected:near(r['delivered_export_mwh'],r['authorized_export_mwh']-r['idle_authorization_mwh'],1e-8)
            t={k:sum(r[k] for r in selected) for k in FIELDS}
            totals[name][c]=t
            if name!='PV100':
                for k in FIELDS:near(t[k],v['totals'][name][c][k],1e-8)
            else:
                for k,pk in [('delivered_export_mwh','delivered'),('authorized_export_mwh','authorized'),('idle_authorization_mwh','unused')]:near(t[k],primary['totals'][c][pk],1e-8)
            energy.append(dict(setting=name,pv_factor=s['factor'],primary=s['is_primary'],control=c,participant_capacity_MW=capacity,
                **t,control_intervals=3840,failed_control_intervals=sum(key[0]==name and key[3]==c for key in failure_set)))
        ecf=np.array([index[name,d,'ECF']['delivered_export_mwh'] for d in p['dates']])
        for c in ['NO_FEEDBACK','MATCHED_UNIFORM','PERSISTENCE_ONLY']:
            diff=ecf-np.array([index[name,d,c]['delivered_export_mwh'] for d in p['dates']])
            mean=float(diff.mean());ci=np.quantile(mean+weights@(diff-mean)/80,[.025,.975])
            dx=totals[name]['ECF']['authorized_export_mwh']-totals[name][c]['authorized_export_mwh']
            dh=totals[name]['ECF']['idle_authorization_mwh']-totals[name][c]['idle_authorization_mwh']
            dy=float(diff.sum());near(dy,dx-dh,1e-8)
            if name!='PV100':
                recorded=v['contrasts'][name+'_minus_'+c]
                near(dy,recorded['total_MWh'],1e-8);near(ci,recorded['date_dependent_95_MWh'],1e-8)
            contrasts.append(dict(setting=name,pv_factor=s['factor'],primary=s['is_primary'],comparator=c,
                delivery_difference_MWh=dy,mean_daily_kWh=mean*1000,CI95_low_daily_kWh=float(ci[0]*1000),CI95_high_daily_kWh=float(ci[1]*1000),
                authorization_difference_MWh=dx,unused_difference_MWh=dh,
                gain_MWh_per_installed_MW=dy/capacity,
                gain_fraction_of_control_delivery=dy/totals[name][c]['delivered_export_mwh'] if totals[name][c]['delivered_export_mwh']>1e-10 else None,
                positive_days=int((diff>1e-10).sum()),negative_days=int((diff<-1e-10).sum()),zero_days=int((np.abs(diff)<=1e-10).sum()),
                qualifying_intervals=totals[name]['ECF']['gate_count'],assessed_intervals=3840,
                failed_control_intervals=sum(key[0]==name and key[3] in ('ECF',c) for key in failure_set),
                failed_replay_intervals=sum(key[0]==name and key[3]=='REPLAY' for key in failure_set) if name!='PV100' else None))
        exposures.append(dict(setting=name,pv_factor=s['factor'],primary=s['is_primary'],**exp))
    assert len(energy)==20 and len(contrasts)==15 and len(exposures)==5
    for r in contrasts:
        assert r['positive_days']+r['negative_days']+r['zero_days']==80
        assert np.isfinite([r['mean_daily_kWh'],r['CI95_low_daily_kWh'],r['CI95_high_daily_kWh']]).all()
        assert r['CI95_low_daily_kWh']<=r['mean_daily_kWh']+1e-10<=r['CI95_high_daily_kWh']+1e-10
    for name,h in source_hashes.items():assert run.sha(Path(name))==h,name
    run.verify_sources()
    output=dict(settings=settings,energy=energy,contrasts=contrasts,exposure=exposures,daily=daily,
        unique_failed_records=[dict(setting=m,date=d,interval=t,control=c) for m,d,t,c in sorted(failure_set)],
        counts=dict(new_days=320,reference_days=80,new_control_intervals=61440,reference_control_intervals=15360,
                    all_control_intervals=76800,new_independently_checked_replays=15360),
        interval_method='Pointwise paired-day dependent wild bootstrap, 10000 replicates, calendar kernel 8 days, key 184. The primary reference uses this same sensitivity-family calculation; existing primary manuscript intervals are not replaced.',
        denominator_scope='Undelivered available export is not entirely physically recoverable and is not a causal opportunity set.',
        source_hashes=source_hashes,producer_sha256=run.sha(Path(__file__)),manuscript_changed=False)
    (A/'PUBLICATION_DATA.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(ready=True,energy_rows=len(energy),contrast_rows=len(contrasts),daily_rows=len(daily),unique_failures=len(failure_set))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check-prerequisites',action='store_true');args=ap.parse_args()
    if args.check_prerequisites:print(json.dumps(prerequisites()))
    else:main()
