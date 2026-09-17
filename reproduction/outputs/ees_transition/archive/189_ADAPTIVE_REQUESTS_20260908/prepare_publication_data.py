"""Aggregate the complete, audited behavior sensitivity without pseudo-replication.

No manuscript or figure writes. Incomplete input is rejected. Failed physical
records are carried into the output, never excluded to improve a contrast.
"""
from pathlib import Path
import argparse,json
import numpy as np
from run_adaptation import A,BRANCHES,verify,sha,read,write

MODES=['ADAPTIVE','STATIC_RANDOM'];SEEDS=[2001,2002,2003]
CONTROLS=BRANCHES+['MATCHED_UNIFORM_SHADOW']
MEASURES={'authorization_mwh':'authorized_export_kw','delivery_mwh':'delivered_export_kw',
          'idle_mwh':'idle_authorization_kw','source_exchange_mwh':'actual_source_terminal_p_kw'}

def day_weights(dates,seed=189):
    days=np.asarray(dates,dtype='datetime64[D]').astype(int)
    kernel=np.maximum(1-np.abs(days[:,None]-days[None,:])/8,0)
    eig,rot=np.linalg.eigh(kernel)
    assert eig.min()>-1e-10
    half=np.random.default_rng(seed).standard_normal((5000,len(days)))@(rot*np.sqrt(np.maximum(eig,0))).T
    return np.concatenate([half,-half])

def describe(diff,weights):
    diff=np.asarray(diff,dtype=float);assert diff.ndim==1 and np.isfinite(diff).all()
    mean=float(diff.mean())
    ci=np.quantile(mean+weights@(diff-mean)/len(diff),[.025,.975])
    return {'total_MWh':float(diff.sum()),'mean_daily_kWh':mean*1000,
            'CI95_low_daily_kWh':float(ci[0]*1000),'CI95_high_daily_kWh':float(ci[1]*1000),
            'positive_dates':int((diff>1e-10).sum()),'negative_dates':int((diff<-1e-10).sum()),
            'zero_dates':int((np.abs(diff)<=1e-10).sum())}

def summarize(rows,dates,blocks):
    ix={(q['mode'],q['seed'],q['date'],q['control']):q for q in rows}
    expected={(m,s,d,c) for m in MODES for s in SEEDS for d in dates for c in CONTROLS}
    assert len(ix)==len(rows)==len(expected) and set(ix)==expected,'Missing or duplicate setting-date-control'
    weights=day_weights(dates);series={};totals=[];comparisons=[];block_results=[]
    for mode in MODES:
        for control in CONTROLS:
            data={k:np.array([[ix[mode,s,d,control][k] for d in dates] for s in SEEDS]) for k in MEASURES}
            series[mode,control]=data
            for si,seed in enumerate(SEEDS):
                totals.append(dict(mode=mode,control=control,seed=seed,**{k:float(v[si].sum()) for k,v in data.items()}))
            totals.append(dict(mode=mode,control=control,seed='SEED_MEAN',**{k:float(v.mean(axis=0).sum()) for k,v in data.items()}))
        for control in ['NO_FEEDBACK','UNQUALIFIED_SAME_UPDATE','MATCHED_UNIFORM_SHADOW']:
            deltas={k:series[mode,'ECF'][k]-series[mode,control][k] for k in MEASURES}
            assert np.max(np.abs(deltas['delivery_mwh']-(deltas['authorization_mwh']-deltas['idle_mwh'])))<1e-9
            for si,seed in list(enumerate(SEEDS))+[(None,'SEED_MEAN')]:
                get=lambda k:deltas[k].mean(axis=0) if si is None else deltas[k][si]
                diff=get('delivery_mwh');baseline=series[mode,control]['delivery_mwh'].mean(axis=0) if si is None else series[mode,control]['delivery_mwh'][si]
                comparisons.append(dict(mode=mode,control=control,seed=seed,**describe(diff,weights),
                    authorization_difference_MWh=float(get('authorization_mwh').sum()),
                    unused_difference_MWh=float(get('idle_mwh').sum()),
                    source_exchange_difference_MWh=float(get('source_exchange_mwh').sum()),
                    percent_of_control_delivery=100*float(diff.sum())/float(baseline.sum()) if baseline.sum()>0 else None))
                for bi,bd in enumerate(blocks):
                    ids=[dates.index(d) for d in bd]
                    block_results.append(dict(mode=mode,control=control,seed=seed,block=bi+1,dates=len(ids),total_MWh=float(diff[ids].sum())))
    interaction=[]
    for control in ['NO_FEEDBACK','UNQUALIFIED_SAME_UPDATE','MATCHED_UNIFORM_SHADOW']:
        def contrast(mode):return series[mode,'ECF']['delivery_mwh']-series[mode,control]['delivery_mwh']
        change=contrast('ADAPTIVE')-contrast('STATIC_RANDOM')
        for si,seed in list(enumerate(SEEDS))+[(None,'SEED_MEAN')]:
            diff=change.mean(axis=0) if si is None else change[si]
            interaction.append(dict(control=control,seed=seed,**describe(diff,weights)))
    return dict(energy_totals=totals,ecf_contrasts=comparisons,change_in_ecf_contrast_due_to_preference_updates=interaction,block_contrasts=block_results)

def prerequisites():
    names=['PIPELINE_COMPLETE.json','VERIFIED_ADAPTATION.json','VERIFIED_SOURCE_RECORDS.json','VERIFIED_SAVED_AC.json','SAVED_AC_TESTS.json','POSTPROCESS_TESTS.json','SAVED_AUDIT_TESTS.json']
    missing=[n for n in names if not (A/n).exists()]
    if missing:return {'ready':False,'missing':missing,'publication_data_written':False}
    p=verify();v=read(A/'VERIFIED_ADAPTATION.json');end=read(A/'PIPELINE_COMPLETE.json');test=read(A/'POSTPROCESS_TESTS.json')
    assert end['audit_sha256']==sha(A/'VERIFIED_ADAPTATION.json') and end['protocol_sha256']==sha(A/'PROTOCOL.json')
    assert v['complete'] and not v['smoke'] and v['verified_setting_days']==480
    assert v['participant_updates_checked']==480*48*418*3 and v['replays_checked']==480*48
    source=read(A/'VERIFIED_SOURCE_RECORDS.json')
    assert source['complete'] and source['verified_setting_days']==480
    assert source['protocol_sha256']==sha(A/'PROTOCOL.json') and source['checker_sha256']==sha(A/'check_source_records.py')
    assert source['files']==v['files']
    ac=read(A/'VERIFIED_SAVED_AC.json');ac_test=read(A/'SAVED_AC_TESTS.json')
    assert ac['complete'] and ac['setting_days']==480 and ac['decision_states']==115200
    assert not ac['contradictions'] and ac['protocol_sha256']==sha(A/'PROTOCOL.json')
    assert ac['checker_sha256']==sha(A/'check_saved_ac.py') and ac_test['passed']
    assert ac_test['checker_sha256']==ac['checker_sha256'] and ac_test['test_sha256']==sha(A/'test_saved_ac.py')
    assert len(ac['report_files'])==480
    ac_inputs={}
    for path,h in ac['report_files'].items():
        assert sha(path)==h,path
        report=read(path)
        assert report['checker_sha256']==ac['checker_sha256'] and report['protocol_sha256']==ac['protocol_sha256']
        assert len(report['screens'])==240 and not report['contradictions']
        assert len({(q['interval'],q['control']) for q in report['screens']})==240
        for n,h in report['files'].items():ac_inputs[str(A/'jobs'/report['job']/report['date']/n)]=h
    assert ac_inputs==v['files']
    faults=read(A/'SAVED_AUDIT_TESTS.json')
    assert faults['passed'] and len(faults['faults_rejected'])==15
    assert faults['protocol_sha256']==sha(A/'PROTOCOL.json') and faults['auditor_sha256']==v['auditor_sha256']
    assert faults['test_sha256']==sha(A/'test_saved_audit.py')
    assert test['passed'] and test['producer_sha256']==sha(Path(__file__))
    for path,h in v['files'].items():assert sha(path)==h,path
    return {'ready':True,'publication_data_written':False}

def main():
    ready=prerequisites();assert ready['ready'],ready
    p=verify();v=read(A/'VERIFIED_ADAPTATION.json');rows=[];behavior=[]
    for job in p['jobs']:
        for d in job['dates']:
            out=A/'jobs'/job['id']/d;r=read(out/'RESULT.json');records=r['records']
            for c in CONTROLS:
                qs=[q for q in records if q['control']==c];assert len(qs)==48
                row=dict(mode=job['mode'],seed=job['seed'],block=job['block'],date=d,control=c)
                row.update({k:sum(q[f] for q in qs)*.5/1000 for k,f in MEASURES.items()})
                assert abs(row['delivery_mwh']-(row['authorization_mwh']-row['idle_mwh']))<1e-9
                row['qualified_intervals']=sum(int(q['gate_triggered']) for q in qs)
                row['failed_observed_intervals']=sum(not q['physical_pair_pass'] for q in qs)
                row['failed_replay_intervals']=sum(q['replay_physical_pass'] is False for q in qs)
                rows.append(row)
            with np.load(out/'TRAJECTORIES.npz') as z:
                active=z['base_request_mw']>0
                for c in BRANCHES:
                    probs=z[c+'_probabilities'];action=z[c+'_action'];cap=z['capacity_mw']
                    expected=(probs*np.array([.5,.75,1,1.25,1.5])).sum(axis=-1)
                    behavior.append(dict(mode=job['mode'],seed=job['seed'],block=job['block'],date=d,control=c,
                        positive_base_request_participant_intervals=int(active.sum()),
                        action_counts_among_positive_base_request=[int(((action==i)&active).sum()) for i in range(5)],
                        expected_multiplier_mean=float(expected[active].mean()) if active.any() else None,
                        raw_request_MWh=float(z[c+'_raw'].sum()*.5),
                        admitted_request_MWh=float(z[c+'_request'].sum()*.5),
                        binding_ceiling_participant_intervals=int(((z[c+'_raw']-z[c+'_request'])>2e-6).sum()),
                        end_ceiling_mean_fraction_of_capacity=float((z[c+'_end_ceiling']/cap).mean()),
                        mean_absolute_end_weight_difference_from_start=float(np.abs(z[c+'_end_weights']-z[c+'_start_weights']).mean())))
    summary=summarize(rows,p['dates'],p['blocks'])
    output=dict(**summary,daily=rows,behavior_diagnostics=behavior,physical_failures=v['physical_failures'],
        unique_observed_dates=80,setting_days=480,control_interval_records=92160,replay_intervals=23040,
        uncertainty='Pointwise paired-date dependent wild intervals: eight-day calendar Bartlett kernel, 10000 antithetic draws, seed 189. Seed means are formed within each original date before intervals. No multiple-comparison correction or independent-feeder inference.',
        shadow_scope='Equal-total shadow follows ECF raw requests and history; it is not an independently learning uniform controller.',
        adaptation_scope='Own completed-delivery reward updates action preferences. Not foresighted strategy, incentive compatibility, field-calibrated behavior, or a transferred bandit regret guarantee.',
        source_hashes={str(A/n):sha(A/n) for n in ['PROTOCOL.json','VERIFIED_ADAPTATION.json','VERIFIED_SOURCE_RECORDS.json','VERIFIED_SAVED_AC.json','SAVED_AC_TESTS.json','PIPELINE_COMPLETE.json','POSTPROCESS_TESTS.json','SAVED_AUDIT_TESTS.json']},
        producer_sha256=sha(Path(__file__)),manuscript_changed=False)
    write(A/'PUBLICATION_DATA.json',output)
    print(json.dumps({'daily_rows':len(rows),'behavior_rows':len(behavior),'contrast_rows':len(summary['ecf_contrasts']),'physical_failures':len(v['physical_failures']),'manuscript_changed':False}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check-prerequisites',action='store_true');a=ap.parse_args()
    if a.check_prerequisites:print(json.dumps(prerequisites()))
    else:main()
